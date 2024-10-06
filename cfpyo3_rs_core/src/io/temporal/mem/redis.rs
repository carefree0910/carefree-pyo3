use super::{Fetcher, FetcherArgs};
#[cfg(feature = "bench-io-mem-redis")]
use crate::toolkit::misc::{Statics, Trackers};
use crate::toolkit::{
    array::AFloat,
    convert::{from_bytes, to_nbytes},
};
use anyhow::{anyhow, Context, Result};
use numpy::{
    ndarray::{Array1, ArrayView1, CowArray},
    Ix1, PyFixedString,
};
use redis::{
    cluster::{ClusterClient, ClusterClientBuilder, ClusterConnection},
    Commands, Script,
};
use std::{env, iter::zip, marker::PhantomData, sync::Mutex};

// core implementations

#[derive(Debug)]
struct RedisError(String);
impl RedisError {
    pub fn data_not_enough<T>(ctx: &str) -> Result<T> {
        Err(RedisError(format!("fetched data is not enough ({})", ctx)).into())
    }
}
impl std::error::Error for RedisError {}
impl std::fmt::Display for RedisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error occurred in `redis` module: {}", self.0)
    }
}

pub const REDIS_KEY_NBYTES: usize = 256;
pub type RedisKey = PyFixedString<REDIS_KEY_NBYTES>;
struct Cursor(usize);
pub struct RedisClient<T: AFloat> {
    cursor: Mutex<Cursor>,
    cluster: Option<Mutex<ClusterClient>>,
    conn_pool: Option<Vec<Option<Mutex<ClusterConnection>>>>,
    warmed_up: bool,
    phantom: PhantomData<T>,
    #[cfg(feature = "bench-io-mem-redis")]
    trackers: Trackers,
}

fn init_client(urls: Vec<String>) -> Result<Mutex<ClusterClient>> {
    let username = env::var("REDIS_USER").unwrap_or("default".to_string());
    let password = env::var("REDIS_PASSWORD").unwrap_or("".to_string());
    let connection_timeout = env::var("REDIS_CONNECTION_TIMEOUT")
        .unwrap_or("30".to_string())
        .parse::<u64>()
        .expect("failed to parse REDIS_CONNECTION_TIMEOUT from env");
    let connection_timeout = std::time::Duration::from_secs(connection_timeout);
    let client = ClusterClientBuilder::new(urls.clone())
        .username(username)
        .password(password)
        .connection_timeout(connection_timeout)
        .response_timeout(connection_timeout)
        .build()?;
    Ok(Mutex::new(client))
}

fn roll_pool_idx(cursor: &Mutex<Cursor>, pool: &[Option<Mutex<ClusterConnection>>]) -> usize {
    let mut cursor = cursor.lock().unwrap();
    let current = cursor.0;
    for i in 0..pool.len() {
        let idx = (current + i) % pool.len();
        if pool[idx].as_ref().unwrap().try_lock().is_ok() {
            cursor.0 = (idx + 1) % pool.len();
            return idx;
        }
    }
    cursor.0 = (current + 1) % pool.len();
    current
}

impl<T: AFloat> RedisClient<T> {
    pub fn new() -> Self {
        Self {
            cursor: Mutex::new(Cursor(0)),
            cluster: None,
            conn_pool: None,
            warmed_up: false,
            phantom: PhantomData::<T>,
            #[cfg(feature = "bench-io-mem-redis")]
            trackers: Trackers::new(0),
        }
    }

    pub fn reset(&mut self, urls: Vec<String>, pool_size: usize, reconnect: bool) -> Result<()> {
        if reconnect || self.cluster.is_none() {
            self.cluster = Some(init_client(urls)?);
        }
        if reconnect || self.conn_pool.is_none() {
            self.conn_pool = Some((0..pool_size).map(|_| None).collect::<Vec<_>>());
            #[cfg(feature = "bench-io-mem-redis")]
            {
                self.trackers = Trackers::new(pool_size);
            }
            match (&mut self.cluster, &mut self.conn_pool) {
                (Some(cluster), Some(pool)) => {
                    pool.iter_mut().try_for_each(|conn| {
                        if conn.is_none() {
                            let new_conn = cluster.get_mut().unwrap().get_connection();
                            match new_conn {
                                Ok(new_conn) => {
                                    *conn = Some(Mutex::new(new_conn));
                                }
                                Err(err) => {
                                    return Err(err).with_context(|| {
                                        "failed to execute `get_connection` from cluster"
                                    });
                                }
                            }
                        }
                        Ok(())
                    })?;
                    self.warmed_up = true;
                }
                _ => {
                    return Err(anyhow!(
                        "internal error: `cluster` or `conn_pool` is `None`"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn fetch(&self, key: &str, start: isize, end: isize) -> Result<Array1<T>> {
        if !self.warmed_up {
            panic!("should call `reset` before `fetch`");
        }
        match &self.conn_pool {
            None => panic!("should call `reset` before `fetch`"),
            Some(pool) => {
                let idx = roll_pool_idx(&self.cursor, pool);
                let rv: Vec<u8> = {
                    let mut conn = pool[idx].as_ref().unwrap().lock().unwrap();
                    #[cfg(feature = "bench-io-mem-redis")]
                    {
                        let now = std::time::Instant::now();
                        let rv: Vec<u8> = conn.getrange(key, start, end - 1)?;
                        self.trackers.track(idx, now.elapsed().as_secs_f64());
                        rv
                    }
                    #[cfg(not(feature = "bench-io-mem-redis"))]
                    {
                        conn.getrange(key, start, end - 1)?
                    }
                };
                if rv.len() != (end - start) as usize {
                    return RedisError::data_not_enough(&format!(
                        "key: {}, start: {}, end: {}; fetched: {}, expected: {}",
                        key,
                        start,
                        end,
                        rv.len(),
                        end - start,
                    ));
                }
                Ok(Array1::from(unsafe { from_bytes(rv) }))
            }
        }
    }

    pub fn batch_fetch(
        &self,
        key: &str,
        start_indices: Vec<isize>,
        end_indices: Vec<isize>,
    ) -> Result<Vec<Array1<T>>> {
        if !self.warmed_up {
            panic!("should call `reset` before `batch_fetch`");
        }
        if start_indices.len() != end_indices.len() {
            panic!("`start_indices` & `end_indices` should have the same length");
        }
        match &self.conn_pool {
            None => panic!("should call `reset` before `batch_fetch`"),
            Some(pool) => {
                let script = Script::new(
                    r#"
                        local key = KEYS[1]
                        local results = {}
                        for i, range in ipairs(ARGV) do
                            local start, end_ = range:match("(%d+)-(%d+)")
                            results[i] = redis.call("GETRANGE", key, start, end_ - 1)
                        end
                        return results
                    "#,
                );
                let ranges_str: Vec<String> = zip(&start_indices, &end_indices)
                    .map(|(start, end)| format!("{}-{}", start, end))
                    .collect();
                let rv: Vec<Vec<u8>> = {
                    let idx = roll_pool_idx(&self.cursor, pool);
                    let mut conn = pool[idx].as_ref().unwrap().lock().unwrap();
                    #[cfg(feature = "bench-io-mem-redis")]
                    {
                        let now = std::time::Instant::now();
                        let rv: Vec<Vec<u8>> =
                            script.key(key).arg(ranges_str).invoke(&mut *conn)?;
                        self.trackers.track(idx, now.elapsed().as_secs_f64());
                        rv
                    }
                    #[cfg(not(feature = "bench-io-mem-redis"))]
                    {
                        script.key(key).arg(ranges_str).invoke(&mut *conn)?
                    }
                };
                if rv.len() != start_indices.len() {
                    return RedisError::data_not_enough(&format!(
                        "key: {}, fetched (outer): {}, expected (outer): {}",
                        key,
                        rv.len(),
                        start_indices.len(),
                    ));
                }
                for (i, i_rv) in rv.iter().enumerate() {
                    if i_rv.len() != (end_indices[i] - start_indices[i]) as usize {
                        return RedisError::data_not_enough(&format!(
                            "key: {}, start: {}, end: {}; fetched (inner): {}, expected (inner): {}",
                            key,
                            start_indices[i],
                            end_indices[i],
                            i_rv.len(),
                            end_indices[i] - start_indices[i],
                        ));
                    }
                }
                Ok(rv
                    .into_iter()
                    .map(|bytes| unsafe { from_bytes(bytes) })
                    .map(Array1::from)
                    .collect())
            }
        }
    }

    #[cfg(feature = "bench-io-mem-redis")]
    pub fn get_tracker_statics(&self) -> Vec<Statics> {
        self.trackers.get_statics()
    }

    #[cfg(feature = "bench-io-mem-redis")]
    pub fn reset_trackers(&self) {
        self.trackers.reset()
    }
}
impl<T: AFloat> Default for RedisClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

// public interface

/// a redis fetcher that makes following assumptions:
/// - a file represents ONE data of ONE day
/// - the data of each day is flattened and concatenated into a single array
///   - it can, however, be either row-contiguous or column-contiguous
pub struct RedisFetcher<'a, T: AFloat> {
    client: &'a RedisClient<T>,
    pub redis_keys: &'a [&'a ArrayView1<'a, RedisKey>],
}

impl<'a, T: AFloat> RedisFetcher<'a, T> {
    pub fn new(client: &'a RedisClient<T>, redis_keys: &'a [&'a ArrayView1<'a, RedisKey>]) -> Self {
        Self { client, redis_keys }
    }
}

impl<'a, T: AFloat> Fetcher<T> for RedisFetcher<'a, T> {
    fn can_batch_fetch(&self) -> bool {
        true
    }

    fn batch_fetch(&self, args: Vec<FetcherArgs>) -> Result<Vec<CowArray<T, Ix1>>> {
        let c = args[0]
            .c
            .expect("`c` should not be `None` in RedisFetcher::batch_fetch");
        let date_idx = args[0].date_idx;
        if args
            .iter()
            .any(|arg| arg.c != Some(c) || arg.date_idx != date_idx)
        {
            panic!("`c` & `date_idx` should be the same in RedisFetcher::batch_fetch");
        }
        let key = self.redis_keys[c][date_idx as usize];
        let key = std::str::from_utf8(&key.0)?.trim_end_matches(char::from(0));
        let args_len = args.len();
        let mut start_indices = Vec::with_capacity(args_len);
        let mut end_indices = Vec::with_capacity(args_len);
        for arg in args {
            start_indices.push(to_nbytes::<T>(arg.time_start_idx as usize) as isize);
            end_indices.push(to_nbytes::<T>(arg.time_end_idx as usize) as isize);
        }
        self.client
            .batch_fetch(key, start_indices, end_indices)?
            .into_iter()
            .map(|array| Ok(array.into()))
            .collect()
    }
}

/// similar to `RedisFetcher`, but it makes slightly different assumptions:
/// - a file represents MULTIPLE data of ONE day (let's say, `C` data)
/// - the `C` dimension is at the last axis, which means it is 'feature-contiguous'
///   - it can, however, be either row-contiguous or column-contiguous for the first axis
///   - it is suggested to use column-contiguous in this case, because the purpose of grouping
///     features together is to make column-contiguous scheme more efficient
pub struct RedisGroupedFetcher<'a, T: AFloat> {
    client: &'a RedisClient<T>,
    pub multiplier: i64,
    pub redis_keys: &'a ArrayView1<'a, RedisKey>,
}

impl<'a, T: AFloat> RedisGroupedFetcher<'a, T> {
    pub fn new(
        client: &'a RedisClient<T>,
        multiplier: i64,
        redis_keys: &'a ArrayView1<'a, RedisKey>,
    ) -> Self {
        Self {
            client,
            multiplier,
            redis_keys,
        }
    }
}

impl<'a, T: AFloat> Fetcher<T> for RedisGroupedFetcher<'a, T> {
    fn fetch(&self, args: FetcherArgs) -> Result<CowArray<T, Ix1>> {
        let key = self.redis_keys[args.date_idx as usize];
        let key = std::str::from_utf8(&key.0)?.trim_end_matches(char::from(0));
        let start = to_nbytes::<T>(args.time_start_idx as usize) * self.multiplier as usize;
        let end = to_nbytes::<T>(args.time_end_idx as usize) * self.multiplier as usize;
        Ok(self.client.fetch(key, start as isize, end as isize)?.into())
    }
}
