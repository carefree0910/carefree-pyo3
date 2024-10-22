use super::misc::init_rt;
use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
};
use tokio::{runtime::Runtime, task::JoinHandle};

pub trait Worker<T, R>: Send + Sync
where
    T: Send + Sync,
    R: Send + Sync,
{
    fn process(&self, cursor: usize, data: T) -> Result<R>;
}

pub struct AsyncQueue<T, R>
where
    T: Send + Sync,
    R: Send + Sync,
{
    rt: Runtime,
    worker: Arc<RwLock<Box<dyn Worker<T, R>>>>,
    results: Arc<Mutex<HashMap<usize, Result<R>>>>,
    pending: Vec<JoinHandle<()>>,
}
impl<T, R> AsyncQueue<T, R>
where
    T: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    pub fn new(worker: Box<dyn Worker<T, R>>, num_threads: usize) -> Result<Self> {
        Ok(Self {
            rt: init_rt(num_threads)?,
            worker: Arc::new(RwLock::new(worker)),
            results: Arc::new(Mutex::new(HashMap::new())),
            pending: Vec::new(),
        })
    }

    pub fn submit(&mut self, cursor: usize, data: T) {
        let worker = Arc::clone(&self.worker);
        let results = Arc::clone(&self.results);
        let handle = self.rt.spawn(async move {
            let result = worker.read().unwrap().process(cursor, data);
            results.lock().unwrap().insert(cursor, result);
        });
        self.pending.push(handle);
    }

    pub fn pop(&self, cursor: usize) -> Option<Result<R>> {
        self.results.lock().unwrap().remove(&cursor)
    }

    pub fn reset(&mut self, block_after_abort: bool) -> Result<()> {
        use anyhow::Ok;

        self.results.lock().unwrap().clear();
        self.pending.drain(..).try_for_each(|handle| {
            handle.abort();
            if block_after_abort {
                self.rt.block_on(handle)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
