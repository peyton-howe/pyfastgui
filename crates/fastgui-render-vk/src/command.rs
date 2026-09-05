use std::sync::{Arc, Mutex};

use fastgui_core::widget::WidgetTree;
use fastgui_core::{CommandReceiver, CommandSender, OneshotSender, Readback};
use winit::event_loop::EventLoopProxy;

use crate::cuda_texture::CudaExportHandles;
use crate::error::VkRendererError;

/// Mutations a `Window` can receive from any Python thread, applied by the render thread
/// once per frame. `SetClearColor` is the M1 proof case.
///
/// CPU viewport frames never flow through this queue — they use `fastgui_core::FrameSlot`
/// (latest-wins). A `Viewport` widget holds that mailbox; `submit_frame` writes it and
/// [`EventWaker::wake`]s the event loop so the idle `Wait` loop actually redraws.
///
/// `CreateCudaSurface` is the one command whose caller needs a synchronous result (a real
/// device pointer), so it carries a [`OneshotSender`]. `viewport_id` selects which composited
/// layer to replace. M3, unverified against real CUDA hardware — see the
/// `fastgui-interop-cuda` and `cuda_texture` module docs.
///
/// `MutateWidgetTree` is a closure instead of one variant per possible widget mutation —
/// `Window.set_content(widget)` and every individual widget setter all just build a small
/// `move |tree| { ... }` closure in `fastgui-py` and send it here.
pub enum Command {
    SetClearColor([f32; 4]),
    CreateCudaSurface {
        viewport_id: u64,
        width: u32,
        height: u32,
        respond: OneshotSender<Result<CudaExportHandles, VkRendererError>>,
    },
    MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>),
}

/// Wakes a `ControlFlow::Wait` event loop from any thread (a Python worker submitting a
/// frame, a widget setter, …). Bound to an [`EventLoopProxy`] once `Window.run` starts;
/// `wake` is a no-op before that, which is fine — queued commands are drained on the first
/// `RedrawRequested`.
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

/// Command sender that also wakes the render thread, so `ControlFlow::Wait` does not sleep
/// through queued mutations or CUDA-surface replies.
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

/// Cross-thread handles the render thread needs to drain incoming commands and publish
/// state back out for synchronous readback (e.g. `window.clear_color`).
pub struct RenderThreadHandles {
    pub commands: CommandReceiver<Command>,
    pub clear_color: Readback<[f32; 4]>,
    pub waker: EventWaker,
}
