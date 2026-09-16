use std::time::{Duration, Instant};

use fastgui_chrome::ChromeRenderer;
use fastgui_core::widget::{WidgetId, WidgetKind, WidgetTree};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Cap on how often `CursorMoved` (while dragging a `Splitter`/`Slider`) renders synchronously --
/// see `fastgui-render-vk::app`'s identical constant. Keeping this on the Metal backend too:
/// a plain `window.request_redraw()` isn't guaranteed to get a turn against a run loop that's
/// busy dispatching a burst of drag events on *any* platform, not just the Windows/DWM case that
/// originally motivated it there.
const DRAG_RENDER_INTERVAL: Duration = Duration::from_millis(1000 / 60);

/// Reserved `region_id` meaning "the whole `DockArea`" -- see `fastgui-render-vk::app`'s
/// identical constant for the full rationale (shared, backend-agnostic docking logic).
const ROOT_REGION_ID: u64 = 0;

/// How close to the window's edge, while dragging a `Panel` title bar, to prefer a whole-
/// `DockArea` split over splitting whatever panel is currently under the cursor -- see
/// `fastgui-render-vk::app`'s identical constant.
const OUTER_EDGE_MARGIN: f32 = 24.0;

use crate::command::{Command, RenderThreadHandles};
use crate::error::MtlRendererError;
use crate::renderer::MetalRenderer;

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("failed to create the window: {0}")]
    Window(#[from] winit::error::OsError),
    #[error("windowing error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("renderer error: {0}")]
    Renderer(#[from] MtlRendererError),
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
    /// just a clear color -- `basic_window.py` -- and we skip chrome rasterization entirely.
    has_widget_content: bool,
    cursor: (f32, f32),
    dragging_slider: Option<WidgetId>,
    dragging_splitter: Option<WidgetId>,
    /// `(title bar's own WidgetId, that Panel's region_id)` while a `Panel` title bar is being
    /// dragged for rearrangement -- see `fastgui-render-vk::app`'s identical field.
    dragging_panel_title: Option<(WidgetId, u64)>,
    hover_region: Option<(u64, fastgui_core::widget::Rect, fastgui_core::widget::DropZone)>,
    dragging_floating_panel: Option<(WidgetId, (f32, f32))>,
    last_drag_render: Instant,
    window: Option<Window>,
    renderer: Option<MetalRenderer>,
    error: Option<RunError>,
}

impl App {
    /// Apply every command queued since the last frame -- see `fastgui-render-vk::app`'s
    /// identical method. No `CreateCudaSurface` arm: see `crate::command::Command`'s doc comment.
    fn drain_commands(&mut self) {
        while let Ok(command) = self.handles.commands.try_recv() {
            match command {
                Command::SetClearColor(color) => {
                    self.clear_color = color;
                    self.handles.clear_color.set(color);
                }
                Command::MutateWidgetTree(mutation) => {
                    mutation(&mut self.widget_tree);
                    self.has_widget_content = true;
                }
            }
        }
    }

    fn upload_active_content(&mut self) -> Result<(), MtlRendererError> {
        if !self.has_widget_content {
            return Ok(());
        }

        let size_changed = self.last_chrome_size != Some((self.width, self.height));
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

    fn tab_bar_clicked_index(&self, bar_id: WidgetId) -> Option<usize> {
        let rect = self.widget_tree.absolute_rect(bar_id)?;
        let Some(WidgetKind::TabBar { titles, .. }) = self.widget_tree.kind(bar_id) else { return None };
        if titles.is_empty() || rect.width <= 0.0 {
            return None;
        }
        let segment_width = rect.width / titles.len() as f32;
        Some((((self.cursor.0 - rect.x) / segment_width) as usize).min(titles.len() - 1))
    }

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

    fn update_dragged_floating_panel(&mut self) {
        let Some((container_id, (offset_x, offset_y))) = self.dragging_floating_panel else { return };
        self.widget_tree.set_position(container_id, self.cursor.0 - offset_x, self.cursor.1 - offset_y);
    }

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

    /// Apply queued commands and (re-)render one frame right now. Unlike
    /// `fastgui-render-vk::app`'s equivalent, there's no swapchain-recreation cost to debounce
    /// around: `MetalRenderer::resize` only updates the `CAMetalLayer`'s `drawableSize` (no GPU
    /// object recreation at all), so `WindowEvent::Resized` applies it immediately instead of
    /// waiting for a settled debounce window.
    fn render_now(&mut self, event_loop: &ActiveEventLoop) {
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

        match MetalRenderer::new(&window, self.width, self.height) {
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
                self.width = size.width;
                self.height = size.height;
                if let Some(renderer) = &mut self.renderer {
                    if let Err(err) = renderer.resize(self.width, self.height) {
                        self.error = Some(err.into());
                        event_loop.exit();
                        return;
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
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
                if dragging
                    && self.last_drag_render.elapsed() >= DRAG_RENDER_INTERVAL {
                        self.render_now(event_loop);
                        self.last_drag_render = Instant::now();
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
                        self.render_now(event_loop);
                    }
                }
            },
            WindowEvent::RedrawRequested => {
                self.render_now(event_loop);
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Open a window titled `title` at `width`x`height`, initialize Metal against it, and block the
/// calling thread rendering until the window is closed -- the Metal counterpart to
/// `fastgui-render-vk::run`.
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
