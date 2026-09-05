use std::time::{Duration, Instant};

use fastgui_chrome::ChromeRenderer;
use fastgui_core::widget::{WidgetId, WidgetKind, WidgetTree};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Cap on how often `CursorMoved` (while dragging a `Splitter`/`Slider`) renders synchronously —
/// see the long comment on that call site for why this can't just be `window.request_redraw()`
/// (starved by `WM_MOUSEMOVE` on Windows). A plain 60fps cap is fine here: dragging only updates
/// a flex ratio and re-rasterizes chrome at the *current* (unchanging-during-a-splitter-drag)
/// window size — profiling during M6 development never observed a single `render_now` call for
/// that case past a few ms (see ROADMAP.md's M6 status).
const DRAG_RENDER_INTERVAL: Duration = Duration::from_millis(1000 / 60);

/// How long to wait, after the *last* `Resized`, before actually recreating the swapchain — i.e.
/// once per resize gesture (mouse pause/release), not once per `WM_SIZE`. This is the one thing
/// that actually fixed live resize (see ROADMAP.md's M6 status for the full investigation): a
/// throwaway standalone probe (`examples/resize_probe.rs`) with finer instrumentation showed each
/// `VulkanRenderer::resize` costs roughly **2 seconds of externally-imposed blocking** on this
/// machine — `RedrawRequested` essentially stops firing for ~2s after every recreation, even
/// though our own Vulkan calls measure at ~20-30ms — almost certainly DWM redoing expensive
/// bookkeeping every time a swapchain is created/destroyed (confirmed real, physical hardware,
/// not a VM; a Notepad window doing the identical `SetWindowPos` burst took 538ms vs. fastgui's
/// tens of seconds — see the M6 status writeup for the full elimination process: not background
/// contention, not incorrect synchronization primitives — though that was a real bug fixed along
/// the way — not present-mode choice, not validation layers, not frame-rate capping). Recreating
/// N times during one drag costs ~N × 2s; debouncing to recreate once, only once the drag
/// actually pauses, is the only lever that changes that multiplier instead of fighting it.
const RESIZE_DEBOUNCE: Duration = Duration::from_millis(150);

/// Reserved `region_id` meaning "the whole `DockArea`", never assigned to a real `Panel`/`Tabs`
/// (`fastgui-py`'s `NEXT_REGION_ID` counter starts at 1) — see `update_panel_drag_hover`'s doc
/// comment for why this exists: without it, there'd be no way to make a dragged panel span the
/// full width/height as a new top/bottom/left/right row/column, only ever split relative to
/// whichever specific panel happens to be under the cursor.
const ROOT_REGION_ID: u64 = 0;

/// How close to the *window's* edge (not a specific panel's) the cursor needs to be, while
/// dragging a `Panel` title bar, to prefer a whole-`DockArea` split over splitting just whatever
/// panel is currently under it. Deliberately much smaller than a typical panel — this only
/// kicks in right at the outer window edge, not "the outer 25% of whatever panel happens to be
/// there" (that's `DropZone::classify`'s job, for the normal per-panel case).
const OUTER_EDGE_MARGIN: f32 = 24.0;

use crate::command::{Command, RenderThreadHandles};
use crate::error::VkRendererError;
use crate::renderer::VulkanRenderer;

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("failed to create the window: {0}")]
    Window(#[from] winit::error::OsError),
    #[error("windowing error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("renderer error: {0}")]
    Renderer(#[from] VkRendererError),
}

struct App {
    title: String,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
    handles: RenderThreadHandles,
    widget_tree: WidgetTree,
    chrome: ChromeRenderer,
    last_chrome_size: Option<(u32, u32)>,
    /// Set by `MutateWidgetTree` (`set_content` / `set_viewport`). When false, the window is
    /// just a clear color — `basic_window.py` — and we skip chrome rasterization entirely.
    has_widget_content: bool,
    cursor: (f32, f32),
    dragging_slider: Option<WidgetId>,
    dragging_splitter: Option<WidgetId>,
    /// `(title bar's own WidgetId, that Panel's region_id)` while a `Panel` title bar is being
    /// dragged for rearrangement — see `handle_mouse_press`'s `PanelTitleBar` arm.
    dragging_panel_title: Option<(WidgetId, u64)>,
    /// Recomputed every `CursorMoved` while `dragging_panel_title` is `Some`: which region (if
    /// any) the cursor is currently over and which `DropZone` within it — drives both the
    /// drop-indicator overlay (`render_now` passes it to `ChromeRenderer::rasterize`) and what
    /// `MouseInput::Released` actually commits.
    hover_region: Option<(u64, fastgui_core::widget::Rect, fastgui_core::widget::DropZone)>,
    /// `(floating panel's own container WidgetId, cursor's offset from that container's
    /// top-left when the drag started)` — set instead of `dragging_panel_title` when the
    /// grabbed `PanelTitleBar` has `floating: true` (see `Window.add_floating_panel`). Moves the
    /// panel directly; never touches `hover_region`/drop-zone logic at all.
    dragging_floating_panel: Option<(WidgetId, (f32, f32))>,
    last_drag_render: Instant,
    /// The size `renderer`'s swapchain was last actually recreated for. Compared against
    /// `width`/`height` at the top of `render_now`, which is how a live `Resized` — which no
    /// longer touches the swapchain directly at all, see that handler's doc comment — eventually
    /// gets applied: once `WM_SIZE` stops flooding the queue, the self-perpetuating
    /// `RedrawRequested` loop gets a turn again and `render_now` catches the mismatch up.
    last_applied_size: (u32, u32),
    /// Set on every `Resized`; `render_now` only actually recreates the swapchain once this long
    /// without a new one — see `RESIZE_DEBOUNCE`'s doc comment for why this is the fix, not an
    /// optimization.
    last_resize_event: Instant,
    window: Option<Window>,
    renderer: Option<VulkanRenderer>,
    error: Option<RunError>,
}

impl App {
    /// Apply every command queued since the last frame. Commands are drained in order, so a
    /// burst of calls between frames just collapses to the latest value — exactly right for
    /// plain state like the clear color. `SetViewport` just swaps in a new frame mailbox to
    /// poll; actual CPU frames are picked up separately in `upload_active_content`.
    fn drain_commands(&mut self) {
        while let Ok(command) = self.handles.commands.try_recv() {
            match command {
                Command::SetClearColor(color) => {
                    self.clear_color = color;
                    self.handles.clear_color.set(color);
                }
                Command::CreateCudaSurface { viewport_id, width, height, respond } => {
                    let result = match &mut self.renderer {
                        Some(renderer) => renderer.create_cuda_surface(viewport_id, width, height),
                        None => Err(VkRendererError::RendererNotReady),
                    };
                    // The caller may have given up waiting (or the channel is otherwise
                    // gone); nothing to do on our end either way.
                    let _ = respond.send(result);
                }
                Command::MutateWidgetTree(mutation) => {
                    mutation(&mut self.widget_tree);
                    self.has_widget_content = true;
                }
            }
        }
    }

    /// Upload chrome (if the widget tree is dirty) and any new `Viewport` frames. Returns
    /// `Err` only on a genuine GPU/driver failure.
    fn upload_active_content(&mut self) -> Result<(), VkRendererError> {
        if !self.has_widget_content {
            return Ok(());
        }

        let size_changed = self.last_chrome_size != Some((self.width, self.height));
        // A panel drag changes `hover_region` (and so the drop-indicator overlay) every
        // `CursorMoved` without marking the tree itself dirty — force re-rasterization
        // for the duration of the drag so the overlay actually tracks the cursor.
        let dragging_panel = self.dragging_panel_title.is_some();
        let chrome_dirty = self.widget_tree.is_dirty() || size_changed || dragging_panel;
        if chrome_dirty {
            self.widget_tree.compute_layout(self.width as f32, self.height as f32);
            let drop_indicator = self.hover_region.map(|(_, rect, zone)| (rect, zone));
            let frame = self.chrome.rasterize(&self.widget_tree, self.width, self.height, drop_indicator);
            self.widget_tree.clear_dirty();
            self.last_chrome_size = Some((self.width, self.height));
            if let Some(renderer) = &mut self.renderer {
                renderer.set_chrome_frame(frame)?;
            }
        }

        let mut live_ids = Vec::new();
        let mut uploads = Vec::new();
        for id in self.widget_tree.walk() {
            let Some(WidgetKind::Viewport { viewport_id, frames }) = self.widget_tree.kind(id) else {
                continue;
            };
            live_ids.push(*viewport_id);
            if let Some(frame) = frames.take_latest() {
                uploads.push((*viewport_id, frame));
            }
        }
        if let Some(renderer) = &mut self.renderer {
            for (viewport_id, frame) in uploads {
                renderer.set_layer_frame(viewport_id, frame)?;
            }
            renderer.retain_layers(&live_ids);
        }
        Ok(())
    }

    fn viewport_draws(&self) -> Vec<(u64, fastgui_core::widget::Rect)> {
        let mut draws = Vec::new();
        if !self.has_widget_content {
            return draws;
        }
        for id in self.widget_tree.walk() {
            let Some(WidgetKind::Viewport { viewport_id, .. }) = self.widget_tree.kind(id) else {
                continue;
            };
            let Some(rect) = self.widget_tree.absolute_rect(id) else { continue };
            if rect.width >= 1.0 && rect.height >= 1.0 {
                draws.push((*viewport_id, rect));
            }
        }
        draws
    }

    /// Left-click (or drag-start) at the current cursor position: click a `Button`, or start
    /// dragging a `Slider` (and immediately jump its value to the click position, matching
    /// standard slider-track-click behavior).
    fn handle_mouse_press(&mut self) {
        let Some(id) = self.widget_tree.hit_test(self.cursor.0, self.cursor.1) else { return };
        let Some(kind) = self.widget_tree.kind(id) else { return };
        match kind {
            WidgetKind::Button { on_click, .. } => {
                if let Some(callback) = on_click.clone() {
                    callback();
                }
            }
            WidgetKind::Slider { .. } => {
                self.dragging_slider = Some(id);
                self.update_dragged_slider();
            }
            WidgetKind::Splitter { .. } => {
                self.dragging_splitter = Some(id);
            }
            WidgetKind::TabBar { panel_ids, .. } => {
                // Selecting and drag-starting both fire on press (matches `PanelTitleBar`, which
                // has no separate click-vs-drag threshold either — see `handle_panel_drop`'s doc
                // comment: a "click" is just a drag that never lands on a valid drop target).
                if let Some(index) = self.tab_bar_clicked_index(id) {
                    if let Some(&panel_id) = panel_ids.get(index) {
                        self.dragging_panel_title = Some((id, panel_id));
                        self.hover_region = None;
                    }
                }
                self.handle_tab_click(id);
            }
            WidgetKind::PanelTitleBar { panel_id, floating, container_id, .. } => {
                if *floating {
                    if let Some(container_id) = container_id {
                        if let Some(rect) = self.widget_tree.absolute_rect(*container_id) {
                            let offset = (self.cursor.0 - rect.x, self.cursor.1 - rect.y);
                            self.dragging_floating_panel = Some((*container_id, offset));
                        }
                    }
                } else {
                    self.dragging_panel_title = Some((id, *panel_id));
                    self.hover_region = None;
                }
            }
            WidgetKind::Container { .. } | WidgetKind::Label { .. } | WidgetKind::Viewport { .. } => {}
        }
    }

    /// Which header segment of a `TabBar` the cursor is currently over (equal-width segments
    /// across the bar's own rect) — shared by `handle_tab_click` (select) and `handle_mouse_press`
    /// (which also uses it to seed a possible drag-out, see `WidgetKind::TabBar`'s doc comment).
    fn tab_bar_clicked_index(&self, bar_id: WidgetId) -> Option<usize> {
        let rect = self.widget_tree.absolute_rect(bar_id)?;
        let Some(WidgetKind::TabBar { titles, .. }) = self.widget_tree.kind(bar_id) else { return None };
        if titles.is_empty() || rect.width <= 0.0 {
            return None;
        }
        let segment_width = rect.width / titles.len() as f32;
        Some((((self.cursor.0 - rect.x) / segment_width) as usize).min(titles.len() - 1))
    }

    /// Switch a `TabBar`'s active tab: flip the newly active content wrapper to `Display::Flex`
    /// and every other one to `Display::None`.
    fn handle_tab_click(&mut self, bar_id: WidgetId) {
        let Some(clicked) = self.tab_bar_clicked_index(bar_id) else { return };
        let Some(WidgetKind::TabBar { active, content_ids, .. }) = self.widget_tree.kind(bar_id) else {
            return;
        };
        if clicked == *active {
            return;
        }
        let content_ids = content_ids.clone();

        self.widget_tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::TabBar { active, .. } = kind {
                *active = clicked;
            }
        });
        for (index, &content_id) in content_ids.iter().enumerate() {
            self.widget_tree.set_display(content_id, index == clicked);
        }

        if let Some(WidgetKind::TabBar { on_select: Some(callback), .. }) = self.widget_tree.kind(bar_id) {
            callback(clicked);
        }
    }

    /// Recompute `hover_region` from the current cursor position — called every `CursorMoved`
    /// while `dragging_panel_title` is active. Excludes the dragged panel's own region (can't
    /// drop a panel onto itself) so `hover_region` is always a legal drop target when `Some`.
    ///
    /// Checks the *window's* outer edge first, before falling back to "which panel is the
    /// cursor over": dropping on a specific panel's own edge (`DropZone::classify`, in the
    /// fallback below) only ever splits *that* panel, which is correct there but means there
    /// was otherwise no way to add a panel as a new row/column spanning the *entire* dock area
    /// — you'd only ever get a strip the width/height of whatever single panel you dropped on.
    /// `Window.set_content`'s `DockArea` handles `ROOT_REGION_ID` as "wrap the whole remaining
    /// tree" instead of "replace this one leaf" — see `python/fastgui/__init__.py`'s
    /// `DockArea._on_rearrange`.
    fn update_panel_drag_hover(&mut self) {
        let Some((_, dragged_region_id)) = self.dragging_panel_title else { return };

        let (width, height) = (self.width as f32, self.height as f32);
        let near_outer_edge = self.cursor.0 <= OUTER_EDGE_MARGIN
            || self.cursor.0 >= width - OUTER_EDGE_MARGIN
            || self.cursor.1 <= OUTER_EDGE_MARGIN
            || self.cursor.1 >= height - OUTER_EDGE_MARGIN;
        if near_outer_edge {
            let window_rect = fastgui_core::widget::Rect { x: 0.0, y: 0.0, width, height };
            let (dist_left, dist_right) = (self.cursor.0, width - self.cursor.0);
            let (dist_top, dist_bottom) = (self.cursor.1, height - self.cursor.1);
            let closest = dist_left.min(dist_right).min(dist_top).min(dist_bottom);
            let zone = if closest == dist_left {
                fastgui_core::widget::DropZone::Left
            } else if closest == dist_right {
                fastgui_core::widget::DropZone::Right
            } else if closest == dist_top {
                fastgui_core::widget::DropZone::Top
            } else {
                fastgui_core::widget::DropZone::Bottom
            };
            self.hover_region = Some((ROOT_REGION_ID, window_rect, zone));
            return;
        }

        self.hover_region = self
            .widget_tree
            .find_region_at(self.cursor.0, self.cursor.1)
            .filter(|(region_id, ..)| *region_id != dragged_region_id)
            .map(|(region_id, rect)| {
                let zone = fastgui_core::widget::DropZone::classify(rect, self.cursor.0, self.cursor.1);
                (region_id, rect, zone)
            });
    }

    /// Commit (or cancel) a `Panel` title-bar drag — or a `TabBar` tab drag-out, which reuses
    /// this same `dragging_panel_title`/`hover_region` machinery (see `WidgetKind::TabBar`'s doc
    /// comment) — on mouse release: if the cursor ended over a legal drop target, invoke the
    /// dragged panel's `on_drop` callback — Python's `DockArea` does the actual tree surgery and
    /// re-attaches via `Window.set_content` (see `fastgui-py`'s `Window.set_content` and
    /// `python/fastgui/__init__.py`'s `DockArea`); this side only ever reports the gesture, never
    /// mutates the tree itself. Always clears the drag state, dropped on a target or not — a
    /// "click" (no movement, never entered a valid `hover_region`) is just a drag that ends here
    /// with `hover` still `None`, so it's a safe no-op.
    fn handle_panel_drop(&mut self) {
        let Some((bar_id, dragged_region_id)) = self.dragging_panel_title.take() else { return };
        let hover = self.hover_region.take();
        let Some((target_region_id, _, zone)) = hover else { return };
        let callback = match self.widget_tree.kind(bar_id) {
            Some(WidgetKind::PanelTitleBar { on_drop: Some(callback), .. }) => Some(callback.clone()),
            Some(WidgetKind::TabBar { panel_ids, on_drop, .. }) => panel_ids
                .iter()
                .position(|&id| id == dragged_region_id)
                .and_then(|index| on_drop[index].clone()),
            _ => None,
        };
        let Some(callback) = callback else { return };
        callback(dragged_region_id, target_region_id, zone);
    }

    /// Move a floating panel to track the cursor, preserving the offset recorded when the drag
    /// started (so it moves *with* the cursor rather than snapping its top-left to it).
    fn update_dragged_floating_panel(&mut self) {
        let Some((container_id, (offset_x, offset_y))) = self.dragging_floating_panel else { return };
        self.widget_tree.set_position(container_id, self.cursor.0 - offset_x, self.cursor.1 - offset_y);
    }

    /// Resize the two panes a `Splitter` divides to match the cursor's current position along
    /// its axis. `first`/`second` are siblings of the bar (all three children of the same
    /// parent `Box`); their combined span (`first`'s leading edge to `second`'s trailing edge)
    /// is the total space available to redistribute — the bar's own thickness stays fixed, so
    /// this fraction is computed across the *whole* span (including the bar) rather than just
    /// the two panes, matching how their `flex_grow` values were originally set as fractions of
    /// the same total.
    fn update_dragged_splitter(&mut self) {
        let Some(bar_id) = self.dragging_splitter else { return };
        let Some(WidgetKind::Splitter { direction, first, second, .. }) = self.widget_tree.kind(bar_id) else {
            return;
        };
        let (direction, first, second) = (*direction, *first, *second);
        let Some(first_rect) = self.widget_tree.absolute_rect(first) else { return };
        let Some(second_rect) = self.widget_tree.absolute_rect(second) else { return };

        let (cursor_pos, start, end) = match direction {
            fastgui_core::widget::SplitDirection::Row => {
                (self.cursor.0, first_rect.x, second_rect.x + second_rect.width)
            }
            fastgui_core::widget::SplitDirection::Column => {
                (self.cursor.1, first_rect.y, second_rect.y + second_rect.height)
            }
        };
        let span = end - start;
        if span <= 0.0 {
            return;
        }
        // Keep both panes at least 5% of the span so neither can be dragged to zero width and
        // become impossible to grab back.
        const MIN_FRACTION: f32 = 0.05;
        let ratio = ((cursor_pos - start) / span).clamp(MIN_FRACTION, 1.0 - MIN_FRACTION);

        self.widget_tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::Splitter { ratio: current, .. } = kind {
                *current = ratio;
            }
        });
        const GROW_SCALE: f32 = 1000.0;
        self.widget_tree.set_flex_grow(first, ratio * GROW_SCALE);
        self.widget_tree.set_flex_grow(second, (1.0 - ratio) * GROW_SCALE);
    }

    fn update_dragged_slider(&mut self) {
        let Some(id) = self.dragging_slider else { return };
        let Some(rect) = self.widget_tree.absolute_rect(id) else { return };
        let fraction = if rect.width > 0.0 { ((self.cursor.0 - rect.x) / rect.width).clamp(0.0, 1.0) } else { 0.0 };

        let mut changed_value = None;
        self.widget_tree.mutate_kind(id, |kind| {
            if let WidgetKind::Slider { value, min, max, .. } = kind {
                *value = *min + fraction * (*max - *min);
                changed_value = Some(*value);
            }
        });

        if let (Some(value), Some(WidgetKind::Slider { on_change: Some(callback), .. })) =
            (changed_value, self.widget_tree.kind(id))
        {
            callback(value);
        }
    }

    /// Set the resize cursor while hovering (or dragging) a `Splitter` bar, restoring the
    /// default cursor everywhere else. Windows only does this automatically for its own
    /// non-client resize borders (the OS window edge itself); our splitter bars are ordinary
    /// client-area content, so nothing shows a resize affordance unless we set it explicitly.
    fn update_cursor_icon(&mut self) {
        let Some(window) = &self.window else { return };
        if self.dragging_panel_title.is_some() || self.dragging_floating_panel.is_some() {
            window.set_cursor(winit::window::CursorIcon::Grabbing);
            return;
        }
        let hovered = self
            .dragging_splitter
            .or_else(|| self.widget_tree.hit_test(self.cursor.0, self.cursor.1));
        let icon = match hovered.and_then(|id| self.widget_tree.kind(id)) {
            Some(WidgetKind::Splitter { direction, .. }) => Some(match direction {
                fastgui_core::widget::SplitDirection::Row => winit::window::CursorIcon::ColResize,
                fastgui_core::widget::SplitDirection::Column => winit::window::CursorIcon::RowResize,
            }),
            Some(WidgetKind::PanelTitleBar { on_drop, floating, .. }) if on_drop.is_some() || *floating => {
                Some(winit::window::CursorIcon::Grab)
            }
            _ => None,
        };
        window.set_cursor(icon.unwrap_or(winit::window::CursorIcon::Default));
    }

    /// Apply queued commands and (re-)render one frame right now, rather than waiting for the
    /// next `RedrawRequested`. `RedrawRequested` alone isn't enough to keep the picture live
    /// during an active drag or an OS-driven window resize: Windows pumps window-border resizes
    /// through a modal loop (`WM_ENTERSIZEMOVE`) that only dispatches messages `winit` forwards
    /// to us synchronously — our own `request_redraw()`-triggers-the-next-frame loop doesn't get
    /// a turn again until the drag ends and that modal loop exits. Calling this directly from
    /// `Resized` and from `CursorMoved` (while a `Splitter`/`Slider` drag is in progress) closes
    /// that gap: every intermediate state gets painted immediately instead of only the final one
    /// on release.
    fn render_now(&mut self, event_loop: &ActiveEventLoop) {
        // Catch the swapchain up to the latest known size — but only once the resize has
        // *settled* (see `RESIZE_DEBOUNCE`'s doc comment for why this debounce, not just a
        // throttle, is what actually fixes live resize). `Resized` itself never touches the
        // swapchain; this is the only place `VulkanRenderer::resize` gets called, and it runs
        // unconditionally (not gated behind any *render* throttle) so that once the debounce
        // window passes, the very next frame — whichever path triggers it — always ends up
        // correctly sized.
        let settled = self.last_resize_event.elapsed() >= RESIZE_DEBOUNCE;
        if settled && self.last_applied_size != (self.width, self.height) {
            if let Some(renderer) = &mut self.renderer {
                if let Err(err) = renderer.resize(self.width, self.height) {
                    self.error = Some(err.into());
                    event_loop.exit();
                    return;
                }
            }
            self.last_applied_size = (self.width, self.height);
        }

        self.drain_commands();
        if let Err(err) = self.upload_active_content() {
            self.error = Some(err.into());
            event_loop.exit();
            return;
        }
        let draws = self.viewport_draws();
        let draw_chrome = self.has_widget_content;
        let clear_color = self.clear_color;
        if let Some(renderer) = &mut self.renderer {
            if let Err(err) = renderer.render_frame(clear_color, draw_chrome, &draws) {
                self.error = Some(err.into());
                event_loop.exit();
            }
        }
    }

    /// `Wait` when idle so we don't spin at tens of thousands of presents/sec. `WaitUntil` only
    /// while a resize debounce is pending, so the swapchain still catches up after the drag
    /// settles without a continuous render loop.
    fn schedule_control_flow(&self, event_loop: &ActiveEventLoop) {
        if self.last_applied_size != (self.width, self.height) {
            let elapsed = self.last_resize_event.elapsed();
            if elapsed < RESIZE_DEBOUNCE {
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + (RESIZE_DEBOUNCE - elapsed),
                ));
                return;
            }
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
        event_loop.set_control_flow(ControlFlow::Wait);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Wait);

        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(f64::from(self.width), f64::from(self.height)));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(err) => {
                self.error = Some(err.into());
                event_loop.exit();
                return;
            }
        };

        match VulkanRenderer::new(&window, self.width, self.height) {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                window.request_redraw();
                self.window = Some(window);
            }
            Err(err) => {
                self.error = Some(err.into());
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // Bookkeeping only — never calls `renderer.resize` (swapchain recreation)
                // synchronously here. Measured directly during M6 development (see ROADMAP.md's
                // M6 status, including a standalone isolated probe that pinned this down
                // precisely): each swapchain recreation costs roughly **2 seconds of
                // externally-imposed blocking** on this machine (almost certainly DWM redoing
                // bookkeeping every time a swapchain is created/destroyed) — not anything
                // measurable inside our own Vulkan calls, and not fixable by throttling render
                // rate or fixing synchronization (both tried, neither helped; a real
                // synchronization bug *was* found and fixed along the way, just not the cause of
                // this). Recreating once per `WM_SIZE` during a drag multiplies that ~2s by
                // however many resize events the drag generates — the actual fix is `render_now`'s
                // `RESIZE_DEBOUNCE` check (see its doc comment): only ever recreate once per
                // resize *gesture*, after the drag settles, not once per event. Until then the
                // window shows a stretched/stale image (`VulkanRenderer::render_frame` always
                // fills whatever the *current* swapchain extent is) — a real visual compromise,
                // but confirmed by a standalone stress test to cut a 10-event drag from ~18.5s to
                // ~370ms of unresponsiveness.
                self.width = size.width;
                self.height = size.height;
                self.last_resize_event = Instant::now();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                let dragging = self.dragging_slider.is_some()
                    || self.dragging_splitter.is_some()
                    || self.dragging_panel_title.is_some()
                    || self.dragging_floating_panel.is_some();
                if self.dragging_slider.is_some() {
                    self.update_dragged_slider();
                }
                if self.dragging_splitter.is_some() {
                    self.update_dragged_splitter();
                }
                if self.dragging_panel_title.is_some() {
                    self.update_panel_drag_hover();
                }
                if self.dragging_floating_panel.is_some() {
                    self.update_dragged_floating_panel();
                }
                self.update_cursor_icon();
                if dragging {
                    // Not `window.request_redraw()`: on Windows that maps to
                    // `RedrawWindow(..., RDW_INTERNALPAINT)`, a `WM_PAINT`-class message, which
                    // Win32 always dispatches at *lower* priority than pending input — while the
                    // mouse is actively moving the queue is never empty of `WM_MOUSEMOVE`, so the
                    // redraw is starved completely until the button comes up. Render
                    // synchronously instead (a real input-adjacent call, not paint-class), but
                    // rate-limited by wall-clock time so a fast drag doesn't force a full render
                    // per pixel of movement — state itself (`update_dragged_*` above) still
                    // applies on every single move regardless, only the paint is capped.
                    if self.last_drag_render.elapsed() >= DRAG_RENDER_INTERVAL {
                        self.render_now(event_loop);
                        self.last_drag_render = Instant::now();
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => match state {
                ElementState::Pressed => self.handle_mouse_press(),
                ElementState::Released => {
                    let was_dragging = self.dragging_slider.is_some()
                        || self.dragging_splitter.is_some()
                        || self.dragging_panel_title.is_some()
                        || self.dragging_floating_panel.is_some();
                    self.dragging_slider = None;
                    self.dragging_splitter = None;
                    self.dragging_floating_panel = None;
                    self.handle_panel_drop();
                    self.update_cursor_icon();
                    if was_dragging {
                        // Paint the final position immediately rather than leaving it to
                        // whenever the starved `RedrawRequested` next gets a turn (see
                        // `CursorMoved`'s comment) — usually fine either way since the message
                        // queue empties out the instant the mouse stops moving, but this makes
                        // release feel crisp instead of possibly-stale for a frame.
                        self.render_now(event_loop);
                    }
                }
            },
            WindowEvent::RedrawRequested => {
                self.render_now(event_loop);
                self.schedule_control_flow(event_loop);
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.schedule_control_flow(event_loop);
    }
}

/// Open a window titled `title` at `width`x`height`, initialize Vulkan against it, and block
/// the calling thread rendering until the window is closed. `handles` lets any other Python
/// thread mutate window state (via its `CommandSender`) and read back the last-applied state
/// (via its `Readback`s) while this call is blocking.
pub fn run(
    title: &str,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
    handles: RenderThreadHandles,
) -> Result<(), RunError> {
    let event_loop = EventLoop::new()?;
    handles.waker.bind(event_loop.create_proxy());
    let mut app = App {
        title: title.to_owned(),
        width,
        height,
        clear_color,
        handles,
        widget_tree: WidgetTree::new(),
        chrome: ChromeRenderer::new(),
        last_chrome_size: None,
        has_widget_content: false,
        cursor: (0.0, 0.0),
        dragging_slider: None,
        dragging_splitter: None,
        dragging_panel_title: None,
        hover_region: None,
        dragging_floating_panel: None,
        last_drag_render: Instant::now(),
        last_applied_size: (width, height),
        last_resize_event: Instant::now(),
        window: None,
        renderer: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}
