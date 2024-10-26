//! # toolkit/queue
//!
//! a module that implements the async queue for lightweight parallel processing.
//!
//! ## Concepts
//!
//! async queue defined here is pretty simple and straightforward. It is a queue that
//! processes the data in parallel using the worker provided by the user, and stores the
//! results in a hashmap. a general workflow of the async queue is as follows:
//!
//! 1. The user creates an async queue with a [`Worker`] and the number of threads, typically
//!    with the [`AsyncQueue::new`] method.
//! 2. The user submits `cursor` and `data` to the async queue with [`AsyncQueue::submit`].
//!    Notice that the `data` should be `Send + Sync` (because it might be used across threads)
//!    and the `cursor` should be unique for each job.
//! 3. The async queue will spawn the `data` to the [`Worker`] asynchronously using the [`Runtime`]
//!    from the [`tokio`] crate.
//! 4. The results from the [`Worker`] are stored in a [`HashMap`] with the `cursor` as the key.
//! 5. The user can then 'poll' the results from the async queue using the `cursor` with the
//!    [`AsyncQueue::pop`] method. The results are removed from the hashmap after the user polls it.
//! 6. After the jobs are done, the user **MUST** call the [`AsyncQueue::reset`] method to
//!    finalize everything properly.

use super::misc::init_rt;
use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
};
use tokio::{runtime::Runtime, task::JoinHandle};

pub trait Worker<T, R>: Send + Sync
where
    R: Send,
{
    fn process(&self, cursor: usize, data: T) -> Result<R>;
}

pub struct AsyncQueue<T, R>
where
    T: Send,
    R: Send,
{
    rt: Runtime,
    worker: Arc<RwLock<Box<dyn Worker<T, R>>>>,
    results: Arc<Mutex<HashMap<usize, Result<R>>>>,
    pending: Vec<JoinHandle<()>>,
}
impl<T, R> AsyncQueue<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
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
