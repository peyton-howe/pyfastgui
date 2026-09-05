pub type CommandSender<T> = crossbeam_channel::Sender<T>;
pub type CommandReceiver<T> = crossbeam_channel::Receiver<T>;

/// An unbounded MPSC channel for funneling mutator calls (from any Python thread) into the
/// single thread that owns the window/renderer/widget tree. Unbounded because a calling
/// thread must never block on the render thread's frame cadence — `Window.set_clear_color()`
/// and friends need to return immediately regardless of how busy the render thread is.
pub fn command_channel<T>() -> (CommandSender<T>, CommandReceiver<T>) {
    crossbeam_channel::unbounded()
}

pub type OneshotSender<T> = crossbeam_channel::Sender<T>;
pub type OneshotReceiver<T> = crossbeam_channel::Receiver<T>;

/// A single-use request/response channel, for the rare mutator that can't just fire-and-forget
/// through `command_channel` because its caller needs a synchronous result back (e.g.
/// `Viewport.create_cuda_surface()`, which must hand back a real device pointer). The calling
/// thread sends a command carrying the `OneshotSender` half, then blocks on the
/// `OneshotReceiver` (via `py.detach()` so it doesn't hold the GIL while waiting) until the
/// render thread replies.
pub fn oneshot_channel<T>() -> (OneshotSender<T>, OneshotReceiver<T>) {
    crossbeam_channel::bounded(1)
}
