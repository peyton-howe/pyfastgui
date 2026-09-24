use std::sync::{Arc, Mutex};

use fastgui_core::widget::WidgetTree;
use fastgui_core::{CommandReceiver, CommandSender, OneshotSender, Readback};
use winit::event_loop::EventLoopProxy;

use crate::cuda_handles::CudaExportHandles;

/// Mutations a `Window` can receive from any Python thread, applied by the render thread.
pub enum Command {
    SetClearColor([f32; 4]),
    /// Vulkan-only. macOS never sends this (Python stubs `create_cuda_surface` first).
    CreateCudaSurface {
        viewport_id: u64,
        width: u32,
        height: u32,
        respond: OneshotSender<Result<CudaExportHandles, String>>,
    },
    MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>),
    /// Widget setter targeting a floating panel's own tree (ids are not unique across trees).
    MutateFloatingTree {
        region_id: u64,
        mutation: Box<dyn FnOnce(&mut WidgetTree) + Send>,
    },
    /// Open `region_id` as a real OS window (not an overlay in the main tree). `build` attaches
    /// the panel as that window's full content. `(x, y, width, height)` are **logical** points,
    /// `(x, y)` relative to the main window's inner origin.
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

/// Command sender that also wakes the render thread -- used by both backends.
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
        Self {
            sender: self.sender.clone(),
            waker: self.waker.clone(),
            floating_region: Some(region_id),
        }
    }

    pub fn send(&self, command: Command) -> Result<(), crossbeam_channel::SendError<Command>> {
        self.sender.send(command)?;
        self.waker.wake();
        Ok(())
    }
}

pub struct RenderThreadHandles {
    pub commands: CommandReceiver<Command>,
    pub clear_color: Readback<[f32; 4]>,
    pub waker: EventWaker,
}
