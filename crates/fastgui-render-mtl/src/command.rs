use std::sync::{Arc, Mutex};

use fastgui_core::widget::WidgetTree;
use fastgui_core::{CommandReceiver, CommandSender, Readback};
use winit::event_loop::EventLoopProxy;

/// Mutations a `Window` can receive from any Python thread, applied by the render thread once
/// per frame. CPU viewport frames use `fastgui_core::FrameSlot` (latest-wins), not this queue.
pub enum Command {
    SetClearColor([f32; 4]),
    MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>),
    /// Widget setter targeting a floating panel's own tree (ids are not unique across trees).
    MutateFloatingTree {
        region_id: u64,
        mutation: Box<dyn FnOnce(&mut WidgetTree) + Send>,
    },
    /// Open `region_id` as a real OS window (not an overlay in the main tree). `build` attaches
    /// the panel as that window's full content. `(x, y)` is the offset from the main window's
    /// inner origin, in the same units as `Window(width, height)`.
    AddFloatingPanel {
        region_id: u64,
        title: String,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        build: Box<dyn FnOnce(&mut WidgetTree) + Send>,
    },
    RemoveFloatingPanel {
        region_id: u64,
    },
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
/// `floating_region` is `Some` for widgets attached inside a floating OS window so later
/// mutators (`label.set_text`, …) hit that window's tree, not the main one (WidgetIds are
/// per-tree and would otherwise collide).
#[derive(Clone)]
pub struct CommandDispatch {
    pub sender: CommandSender<Command>,
    pub waker: EventWaker,
    pub floating_region: Option<u64>,
}

impl CommandDispatch {
    pub fn for_floating(&self, region_id: u64) -> Self {
        Self { sender: self.sender.clone(), waker: self.waker.clone(), floating_region: Some(region_id) }
    }

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
