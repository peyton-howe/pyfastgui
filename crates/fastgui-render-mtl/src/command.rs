use std::sync::{Arc, Mutex};

use fastgui_core::widget::WidgetTree;
use fastgui_core::{CommandReceiver, CommandSender, Readback};
use winit::event_loop::EventLoopProxy;

/// Mutations a `Window` can receive from any Python thread, applied by the render thread once
/// per frame. CPU viewport frames use `fastgui_core::FrameSlot` (latest-wins), not this queue.
pub enum Command {
    SetClearColor([f32; 4]),
    MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>),
}

/// Wakes a `ControlFlow::Wait` event loop from any thread -- see
/// `fastgui-render-vk::EventWaker`, which this mirrors exactly.
#[derive(Clone, Default)]
pub struct EventWaker {
    proxy: Arc<Mutex<Option<EventLoopProxy<()>>>>,
}

impl EventWaker {
    pub fn bind(&self, proxy: EventLoopProxy<()>) {
        *self.proxy.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(proxy);
    }

    pub fn wake(&self) {
        if let Some(proxy) = self.proxy.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone() {
            let _ = proxy.send_event(());
        }
    }
}

/// Command sender that also wakes the render thread -- see `fastgui-render-vk::CommandDispatch`.
#[derive(Clone)]
pub struct CommandDispatch {
    pub sender: CommandSender<Command>,
    pub waker: EventWaker,
}

impl CommandDispatch {
    pub fn send(&self, command: Command) -> Result<(), crossbeam_channel::SendError<Command>> {
        self.sender.send(command)?;
        self.waker.wake();
        Ok(())
    }
}

/// Cross-thread handles the render thread needs to drain incoming commands and publish state
/// back out for synchronous readback -- see `fastgui-render-vk::RenderThreadHandles`.
pub struct RenderThreadHandles {
    pub commands: CommandReceiver<Command>,
    pub clear_color: Readback<[f32; 4]>,
    pub waker: EventWaker,
}
