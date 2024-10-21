#[cfg(feature = "tokio")]
use anyhow::Result;
use md5::{Digest, Md5};
use std::{collections::HashMap, fmt, sync::RwLock, time::Instant};
#[cfg(feature = "tokio")]
use tokio::runtime::{Builder, Runtime};

pub fn hash_code(code: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(code.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// A simple tracker to record the time of each event.
#[derive(Clone)]
pub struct Tracker {
    history: Vec<f64>,
    tracking: Option<Instant>,
}
pub struct Stats {
    n: usize,
    mean: f64,
    std: f64,
    is_fast_path: bool,
    is_bottleneck: bool,
}
impl Tracker {
    pub const fn new() -> Self {
        Self {
            history: Vec::new(),
            tracking: None,
        }
    }

    pub fn track(&mut self, time: f64) {
        self.history.push(time);
    }

    pub fn track_start(&mut self) {
        self.tracking = Some(Instant::now());
    }

    pub fn track_end(&mut self) {
        let time = self
            .tracking
            .expect("please call `track_start` before `track_end`")
            .elapsed()
            .as_secs_f64();
        self.history.push(time);
        self.tracking = None;
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.tracking = None;
    }

    pub fn get_stats(&self) -> Stats {
        let n = self.history.len();
        let mean = self.history.iter().sum::<f64>() / n as f64;
        let variance = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();
        Stats {
            n,
            mean,
            std,
            is_fast_path: false,
            is_bottleneck: false,
        }
    }
}
impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}
impl fmt::Debug for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sum = self.n as f64 * self.mean;
        let prefix = if self.is_bottleneck {
            format!("[🚨 {:.8}] ", sum)
        } else if self.is_fast_path {
            format!("[⚡️ {:.8}] ", sum)
        } else {
            format!("[   {:.8}] ", sum)
        };
        write!(
            f,
            "{}{:.8} ± {:.8} ({})",
            prefix, self.mean, self.std, self.n
        )
    }
}

/// A simple container of `Tracker`s, useful if you want to inspect concurrent events.
pub struct Trackers(pub Vec<RwLock<Tracker>>);
impl Trackers {
    pub fn new(n: usize) -> Self {
        Self((0..n).map(|_| RwLock::new(Tracker::new())).collect())
    }

    pub fn track(&self, idx: usize, time: f64) {
        self.0[idx].write().unwrap().track(time);
    }

    pub fn track_start(&self, idx: usize) {
        self.0[idx].write().unwrap().track_start();
    }

    pub fn track_end(&self, idx: usize) {
        self.0[idx].write().unwrap().track_end();
    }

    pub fn reset(&self) {
        self.0
            .iter()
            .for_each(|tracker| tracker.write().unwrap().reset());
    }

    pub fn get_stats(&self) -> Vec<Stats> {
        let mut stats: Vec<Stats> = self
            .0
            .iter()
            .map(|tracker| tracker.read().unwrap().get_stats())
            .collect();
        let mut fast_path_idx = 0;
        let mut bottleneck_idx = 0;
        let mut fast_path_t = stats[0].n as f64 * stats[0].mean;
        let mut bottleneck_t = fast_path_t;
        for (idx, s) in stats.iter().enumerate() {
            let new_t = s.n as f64 * s.mean;
            if new_t > bottleneck_t {
                bottleneck_idx = idx;
                bottleneck_t = new_t;
            } else if new_t < fast_path_t {
                fast_path_idx = idx;
                fast_path_t = new_t;
            }
        }
        stats[fast_path_idx].is_fast_path = true;
        stats[bottleneck_idx].is_bottleneck = true;
        stats
    }
}

/// A simple, named container of `Tracker`s, useful if you want to inspect different events and compare them.
pub struct NamedTrackers(pub HashMap<String, RwLock<Tracker>>);
impl NamedTrackers {
    pub fn new(names: Vec<String>) -> Self {
        Self(
            names
                .into_iter()
                .map(|name| (name, RwLock::new(Tracker::new())))
                .collect(),
        )
    }

    fn get(&self, name: &str) -> &RwLock<Tracker> {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("'{}' not found in current trackers", name))
    }

    pub fn track(&self, name: &str, time: f64) {
        self.get(name).write().unwrap().track(time);
    }

    pub fn track_start(&self, name: &str) {
        self.get(name).write().unwrap().track_start();
    }

    pub fn track_end(&self, name: &str) {
        self.get(name).write().unwrap().track_end();
    }

    pub fn reset(&self) {
        self.0
            .iter()
            .for_each(|(_, tracker)| tracker.write().unwrap().reset());
    }

    pub fn get_stats(&self) -> HashMap<String, Stats> {
        self.0
            .iter()
            .map(|(name, tracker)| (name.clone(), tracker.read().unwrap().get_stats()))
            .collect()
    }
}

// tokio utils

#[cfg(feature = "tokio")]
pub fn init_rt(num_threads: usize) -> Result<Runtime> {
    let rt = if num_threads <= 1 {
        Builder::new_current_thread().enable_all().build()?
    } else {
        Builder::new_multi_thread()
            .worker_threads(num_threads)
            .enable_all()
            .build()?
    };
    Ok(rt)
}
