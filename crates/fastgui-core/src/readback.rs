use std::sync::{Arc, Mutex};

/// State the render thread publishes after applying a command, so a synchronous Python-side
/// getter (e.g. `window.clear_color`) never blocks on the render thread's frame timing —
/// it just reads whatever was last committed.
#[derive(Clone)]
pub struct Readback<T>(Arc<Mutex<T>>);

impl<T: Clone> Readback<T> {
    pub fn new(initial: T) -> Self {
        Self(Arc::new(Mutex::new(initial)))
    }

    pub fn get(&self) -> T {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    pub fn set(&self, value: T) {
        *self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = value;
    }
}
