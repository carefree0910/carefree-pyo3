use anyhow::Result;
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, LazyLock, Mutex, RwLock},
};
use tokio::{
    runtime::{Builder, Runtime},
    task::JoinHandle,
};

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
    worker: Arc<RwLock<Box<dyn Worker<T, R>>>>,
    results: Arc<Mutex<HashMap<usize, Result<R>>>>,
    pending: Vec<JoinHandle<()>>,
    phantom_task_data: PhantomData<T>,
}
impl<T, R> AsyncQueue<T, R>
where
    T: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    pub fn new(worker: Box<dyn Worker<T, R>>) -> Self {
        Self {
            worker: Arc::new(RwLock::new(worker)),
            results: Arc::new(Mutex::new(HashMap::new())),
            pending: Vec::new(),
            phantom_task_data: PhantomData,
        }
    }

    pub fn submit(&mut self, cursor: usize, data: T) {
        let worker = Arc::clone(&self.worker);
        let results = Arc::clone(&self.results);
        let handle = TWO_THREAD_RT.spawn(async move {
            let result = worker.read().unwrap().process(cursor, data);
            results.lock().unwrap().insert(cursor, result);
        });
        self.pending.push(handle);
    }

    pub fn pop(&self, cursor: usize) -> Option<Result<R>> {
        self.results.lock().unwrap().remove(&cursor)
    }

    pub fn reset(&mut self) {
        self.pending.iter().for_each(|handle| {
            handle.abort();
        });
        self.pending.clear();
        self.results.lock().unwrap().clear();
    }
}

static TWO_THREAD_RT: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
});
