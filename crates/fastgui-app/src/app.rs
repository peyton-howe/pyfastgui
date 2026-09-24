use std::collections::HashMap;
use std::time::Instant;

use fastgui_chrome::ChromeRenderer;
use fastgui_core::widget::{WidgetId, WidgetKind, WidgetTree};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId, WindowLevel};

use crate::command::{Command, RenderThreadHandles};
use crate::constants::*;
use crate::coords::scale_rect;
use crate::ghost::build_tear_ghost_tree;
use crate::resize_edge::{classify_float_resize_edge, resize_edge_cursor, ResizeEdge};
use crate::surface::{MainResizePolicy, SurfaceBackend};

/// Active edge-resize of a floating window, tracked in physical screen space so size/position
/// updates stay stable as the window moves under the cursor.
struct FloatingResizeDrag {
    window_id: WindowId,
    edge: ResizeEdge,
    start_outer: PhysicalPosition<i32>,
    start_inner: PhysicalSize<u32>,
    start_cursor_screen: (f64, f64),
}

/// Panel-shaped preview that follows the cursor while tearing a docked panel out.
/// The real panel stays docked until mouse-up; this window is mouse-transparent so the main
/// window keeps receiving the drag.
struct TearGhost<B: SurfaceBackend> {
    window: Window,
    renderer: B,
    grab_offset: (f32, f32),
}

#[derive(Debug, thiserror::Error)]
pub enum RunError<E: std::error::Error + Send + Sync + 'static> {
    #[error("failed to create the window: {0}")]
    Window(winit::error::OsError),
    #[error("windowing error: {0}")]
    EventLoop(winit::error::EventLoopError),
    #[error("renderer error: {0}")]
    Renderer(E),
}

impl<E: std::error::Error + Send + Sync + 'static> From<winit::error::OsError> for RunError<E> {
    fn from(value: winit::error::OsError) -> Self {
        RunError::Window(value)
    }
}

impl<E: std::error::Error + Send + Sync + 'static> From<winit::error::EventLoopError>
    for RunError<E>
{
    fn from(value: winit::error::EventLoopError) -> Self {
        RunError::EventLoop(value)
    }
}

/// One floating panel as its own undecorated OS window (not an overlay in the main tree).
struct FloatingWindow<B: SurfaceBackend> {
    region_id: u64,
    window: Window,
    renderer: B,
    widget_tree: WidgetTree,
    chrome: ChromeRenderer,
    width: u32,
    height: u32,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    cursor: (f32, f32),
    has_widget_content: bool,
    last_chrome_size: Option<(u32, u32)>,
    last_raster_scale: f64,
    dragging_slider: Option<WidgetId>,
    dragging_splitter: Option<WidgetId>,
}

struct App<B: SurfaceBackend> {
    title: String,
    /// Layout / hit-test size in points (the Python `width`/`height`).
    width: u32,
    height: u32,
    /// Backing-store size in pixels, from winit `Resized` / `inner_size`.
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    clear_color: [f32; 4],
    handles: RenderThreadHandles,
    resize_policy: MainResizePolicy,
    widget_tree: WidgetTree,
    chrome: ChromeRenderer,
    last_chrome_size: Option<(u32, u32)>,
    last_raster_scale: f64,
    /// Set by `MutateWidgetTree` (`set_content` / `set_viewport`). When false, the window is
    /// just a clear color -- `basic_window.py` -- and we skip chrome rasterization entirely.
    has_widget_content: bool,
    cursor: (f32, f32),
    dragging_slider: Option<WidgetId>,
    dragging_splitter: Option<WidgetId>,
    /// `(title bar's own WidgetId, that Panel's region_id)` while a `Panel` title bar is being
    /// dragged for rearrangement.
    dragging_panel_title: Option<(WidgetId, u64)>,
    hover_region: Option<(u64, fastgui_core::widget::Rect, fastgui_core::widget::DropZone)>,
    /// OS-window floater drag: `(floater WindowId, grab offset in that window's logical space)`.
    dragging_floating_panel: Option<(WindowId, (f32, f32))>,
    /// Edge/corner resize of a floating OS window (undecorated — no native resize chrome).
    dragging_floating_resize: Option<FloatingResizeDrag>,
    /// Set with `dragging_panel_title` when a floater title-bar drag also participates in re-dock.
    dragging_from_floating: Option<u64>,
    /// Cursor offset within the dragged docked panel's rect (for tear-off float placement).
    dock_tear_grab: Option<(f32, f32)>,
    /// Size of the docked panel being dragged (becomes the new floating window's size).
    dock_tear_size: Option<(f32, f32)>,
    /// Title captured at docked-panel press — used for the tear-off ghost chrome.
    dock_tear_title: Option<String>,
    /// Main-window cursor at docked-panel press — ghost appears after `TEAR_GHOST_THRESHOLD`.
    dock_tear_press_cursor: Option<(f32, f32)>,
    /// Live ghost preview while tearing a docked panel out (destroyed on mouse-up).
    tear_ghost: Option<TearGhost<B>>,
    /// Region id of a panel just torn out while its ghost is kept on screen. The ghost sits
    /// always-on-top exactly where the new floater opens, so it stays until that floater
    /// presents its first frame instead of leaving a blank window during the handoff.
    tear_handoff: Option<u64>,
    last_drag_render: Instant,
    /// Physical size the main surface was last resized to (Debounced policy).
    last_applied_physical: (u32, u32),
    last_resize_event: Instant,
    window: Option<Window>,
    renderer: Option<B>,
    floating: HashMap<WindowId, FloatingWindow<B>>,
    floating_by_region: HashMap<u64, WindowId>,
    /// `AddFloatingPanel` commands that arrived before `resumed` created the main window.
    pending_floaters: Vec<(
        u64,
        String,
        f32,
        f32,
        f32,
        f32,
        Box<dyn FnOnce(&mut WidgetTree) + Send>,
    )>,
    error: Option<RunError<B::Error>>,
}

impl<B: SurfaceBackend> App<B> {
    /// Apply every command queued since the last frame.
    /// Needs `event_loop` so `AddFloatingPanel` can create additional winit windows.
    fn drain_commands(&mut self, event_loop: &ActiveEventLoop) {
        self.drain_command_queue(event_loop);
        // The float callback sends `AddFloatingPanel` synchronously, so it has been applied by
        // now. If no floater exists for the handoff region, Python declined to float it —
        // don't leave the ghost stranded.
        if let Some(region_id) = self.tear_handoff {
            if !self.floating_by_region.contains_key(&region_id) {
                self.tear_handoff = None;
                self.tear_ghost = None;
            }
        }
    }

    fn drain_command_queue(&mut self, event_loop: &ActiveEventLoop) {
        while let Ok(command) = self.handles.commands.try_recv() {
            match command {
                Command::SetClearColor(color) => {
                    self.clear_color = color;
                    self.handles.clear_color.set(color);
                }
                Command::CreateCudaSurface { viewport_id, width, height, respond } => {
                    let result = match &mut self.renderer {
                        Some(renderer) => renderer.create_cuda_surface(viewport_id, width, height),
                        None => Err("the render thread hasn't finished starting up yet".into()),
                    };
                    let _ = respond.send(result);
                }
                Command::MutateWidgetTree(mutation) => {
                    mutation(&mut self.widget_tree);
                    self.has_widget_content = true;
                }
                Command::MutateFloatingTree { region_id, mutation } => {
                    if let Some(&window_id) = self.floating_by_region.get(&region_id) {
                        if let Some(floater) = self.floating.get_mut(&window_id) {
                            mutation(&mut floater.widget_tree);
                            floater.has_widget_content = true;
                            floater.window.request_redraw();
                        }
                    } else if let Some(pending) =
                        self.pending_floaters.iter_mut().find(|(id, ..)| *id == region_id)
                    {
                        // Window not created yet: run after its `build` instead of dropping it.
                        let build = std::mem::replace(&mut pending.6, Box::new(|_| {}));
                        pending.6 = Box::new(move |tree| {
                            build(tree);
                            mutation(tree);
                        });
                    }
                }
                Command::AddFloatingPanel {
                    region_id,
                    title,
                    x,
                    y,
                    width,
                    height,
                    build,
                } => {
                    self.remove_floating_by_region(region_id);
                    if self.window.is_none() {
                        self.pending_floaters.retain(|(id, ..)| *id != region_id);
                        self.pending_floaters.push((region_id, title, x, y, width, height, build));
                        continue;
                    }
                    if let Err(err) =
                        self.add_floating_panel(event_loop, region_id, title, x, y, width, height, build)
                    {
                        self.error = Some(err);
                        event_loop.exit();
                        return;
                    }
                }
                Command::RemoveFloatingPanel { region_id } => {
                    self.pending_floaters.retain(|(id, ..)| *id != region_id);
                    self.remove_floating_by_region(region_id);
                }
            }
        }
    }

    fn remove_floating_by_region(&mut self, region_id: u64) {
        if let Some(window_id) = self.floating_by_region.remove(&region_id) {
            self.floating.remove(&window_id);
            if self.dragging_from_floating == Some(region_id) {
                self.dragging_from_floating = None;
                self.dragging_panel_title = None;
                self.hover_region = None;
            }
            if self.dragging_floating_panel.is_some_and(|(id, _)| id == window_id) {
                self.dragging_floating_panel = None;
            }
            if self.dragging_floating_resize.as_ref().is_some_and(|d| d.window_id == window_id) {
                self.dragging_floating_resize = None;
            }
        }
    }

    /// Resize edge under a floater's cursor, unless the cursor is on a title-bar × — the
    /// resize strip along the top/right edges would otherwise swallow part of that button.
    fn floating_resize_edge_at_cursor(floater: &FloatingWindow<B>) -> Option<ResizeEdge> {
        let (x, y) = floater.cursor;
        let on_close_button = floater.widget_tree.hit_test(x, y).is_some_and(|id| {
            matches!(
                floater.widget_tree.kind(id),
                Some(WidgetKind::PanelTitleBar { on_close: Some(_), .. })
            ) && floater
                .widget_tree
                .absolute_rect(id)
                .is_some_and(|rect| fastgui_core::widget::close_button_rect(rect).contains(x, y))
        });
        if on_close_button {
            return None;
        }
        classify_float_resize_edge(floater.width as f32, floater.height as f32, x, y)
    }

    /// The close handler on a floater's own title bar, if its panel is closeable.
    fn floating_close_callback(&self, window_id: WindowId) -> Option<fastgui_core::widget::PanelCloseCallback> {
        let floater = self.floating.get(&window_id)?;
        floater.widget_tree.walk().find_map(|id| match floater.widget_tree.kind(id) {
            Some(WidgetKind::PanelTitleBar { panel_id, on_close: Some(callback), .. })
                if *panel_id == floater.region_id =>
            {
                Some(callback.clone())
            }
            _ => None,
        })
    }

    // Mirrors `Command::AddFloatingPanel` field-for-field.
    #[allow(clippy::too_many_arguments)]
    fn add_floating_panel(
        &mut self,
        event_loop: &ActiveEventLoop,
        region_id: u64,
        title: String,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        build: Box<dyn FnOnce(&mut WidgetTree) + Send>,
    ) -> Result<(), RunError<B::Error>> {
        let Some(main) = &self.window else {
            // Caller queues into `pending_floaters` when the main window isn't ready yet.
            return Ok(());
        };
        let scale = main.scale_factor().max(0.01);
        // `(x, y)` is offset from the main window's inner origin in logical points.
        let position = match main.inner_position() {
            Ok(inner) => {
                let logical = inner.to_logical::<f64>(scale);
                LogicalPosition::new(logical.x + f64::from(x), logical.y + f64::from(y))
            }
            Err(_) => LogicalPosition::new(f64::from(x), f64::from(y)),
        };

        let attributes = Window::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(f64::from(width), f64::from(height)))
            .with_position(position)
            .with_decorations(false)
            // Still need custom edge hit-testing (borderless has no OS resize chrome); this
            // flag mainly keeps the window eligible for programmatic `set_inner_size`.
            .with_resizable(true);
        let window = event_loop.create_window(attributes)?;
        let scale_factor = window.scale_factor();
        let physical = window.inner_size();
        let logical = physical.to_logical::<f64>(scale_factor.max(0.01));
        let renderer = B::new(&window, physical.width, physical.height).map_err(RunError::Renderer)?;

        let mut widget_tree = WidgetTree::new();
        build(&mut widget_tree);

        let window_id = window.id();
        let floater = FloatingWindow {
            region_id,
            window,
            renderer,
            widget_tree,
            chrome: ChromeRenderer::new(),
            width: logical.width.round().max(0.0) as u32,
            height: logical.height.round().max(0.0) as u32,
            physical_width: physical.width,
            physical_height: physical.height,
            scale_factor,
            cursor: (0.0, 0.0),
            has_widget_content: true,
            last_chrome_size: None,
            last_raster_scale: 0.0,
            dragging_slider: None,
            dragging_splitter: None,
        };
        floater.window.request_redraw();
        self.floating_by_region.insert(region_id, window_id);
        self.floating.insert(window_id, floater);
        Ok(())
    }

    fn upload_active_content(&mut self) -> Result<(), B::Error> {
        if !self.has_widget_content || self.physical_width == 0 || self.physical_height == 0 {
            return Ok(());
        }

        let size_changed = self.last_chrome_size != Some((self.width, self.height))
            || (self.last_raster_scale - self.scale_factor).abs() > 1e-6;
        let dragging_panel = self.dragging_panel_title.is_some();
        let chrome_dirty = self.widget_tree.is_dirty() || size_changed || dragging_panel;
        if chrome_dirty {
            self.widget_tree.compute_layout(self.width as f32, self.height as f32);
            let drop_indicator = self.hover_region.map(|(_, rect, zone)| (rect, zone));
            let frame = self.chrome.rasterize(
                &self.widget_tree,
                self.physical_width,
                self.physical_height,
                drop_indicator,
                self.scale_factor as f32,
            );
            self.widget_tree.clear_dirty();
            self.last_chrome_size = Some((self.width, self.height));
            self.last_raster_scale = self.scale_factor;
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

    fn upload_floating_content(floater: &mut FloatingWindow<B>) -> Result<(), B::Error> {
        if !floater.has_widget_content || floater.physical_width == 0 || floater.physical_height == 0 {
            return Ok(());
        }

        let size_changed = floater.last_chrome_size != Some((floater.width, floater.height))
            || (floater.last_raster_scale - floater.scale_factor).abs() > 1e-6;
        let chrome_dirty = floater.widget_tree.is_dirty() || size_changed;
        if chrome_dirty {
            floater
                .widget_tree
                .compute_layout(floater.width as f32, floater.height as f32);
            // Drop indicator is drawn on the main window only.
            let frame = floater.chrome.rasterize(
                &floater.widget_tree,
                floater.physical_width,
                floater.physical_height,
                None,
                floater.scale_factor as f32,
            );
            floater.widget_tree.clear_dirty();
            floater.last_chrome_size = Some((floater.width, floater.height));
            floater.last_raster_scale = floater.scale_factor;
            floater.renderer.set_chrome_frame(frame)?;
        }

        let mut live_ids = Vec::new();
        let mut uploads = Vec::new();
        for id in floater.widget_tree.walk() {
            let Some(WidgetKind::Viewport { viewport_id, frames }) = floater.widget_tree.kind(id) else {
                continue;
            };
            live_ids.push(*viewport_id);
            if let Some(frame) = frames.take_latest() {
                uploads.push((*viewport_id, frame));
            }
        }
        for (viewport_id, frame) in uploads {
            floater.renderer.set_layer_frame(viewport_id, frame)?;
        }
        floater.renderer.retain_layers(&live_ids);
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
                draws.push((*viewport_id, scale_rect(rect, self.scale_factor as f32)));
            }
        }
        draws
    }

    fn floating_viewport_draws(floater: &FloatingWindow<B>) -> Vec<(u64, fastgui_core::widget::Rect)> {
        let mut draws = Vec::new();
        if !floater.has_widget_content {
            return draws;
        }
        for id in floater.widget_tree.walk() {
            let Some(WidgetKind::Viewport { viewport_id, .. }) = floater.widget_tree.kind(id) else {
                continue;
            };
            let Some(rect) = floater.widget_tree.absolute_rect(id) else { continue };
            if rect.width >= 1.0 && rect.height >= 1.0 {
                draws.push((*viewport_id, scale_rect(rect, floater.scale_factor as f32)));
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
            WidgetKind::TabBar { panel_ids, content_ids, titles, on_close, .. } => {
                if let Some(index) = self.tab_bar_clicked_index(id) {
                    if let Some(rect) = self.widget_tree.absolute_rect(id) {
                        let n = titles.len().max(1) as f32;
                        let segment = fastgui_core::widget::Rect {
                            x: rect.x + index as f32 * (rect.width / n),
                            y: rect.y,
                            width: rect.width / n,
                            height: rect.height,
                        };
                        if on_close.get(index).is_some_and(|c| c.is_some())
                            && fastgui_core::widget::close_button_rect(segment).contains(self.cursor.0, self.cursor.1)
                        {
                            if let Some(callback) = on_close[index].clone() {
                                if let Some(&panel_id) = panel_ids.get(index) {
                                    callback(panel_id);
                                }
                            }
                            return;
                        }
                    }
                    if let Some(&panel_id) = panel_ids.get(index) {
                        let content_ids = content_ids.clone();
                        let title = titles.get(index).cloned().unwrap_or_default();
                        self.dragging_panel_title = Some((id, panel_id));
                        self.hover_region = None;
                        self.dock_tear_title = Some(title);
                        self.dock_tear_press_cursor = Some(self.cursor);
                        self.seed_dock_tear_from_tab(id, index, &content_ids);
                    }
                }
                self.handle_tab_click(id);
            }
            // Floating panels are real OS windows now — title-bar drag for those is handled on
            // the floater's own `window_event` path, not via `set_position` overlays here.
            WidgetKind::PanelTitleBar { panel_id, floating, title, on_close, .. } => {
                if let Some(rect) = self.widget_tree.absolute_rect(id) {
                    if on_close.is_some()
                        && fastgui_core::widget::close_button_rect(rect).contains(self.cursor.0, self.cursor.1)
                    {
                        if let Some(callback) = on_close.clone() {
                            callback(*panel_id);
                        }
                        return;
                    }
                }
                if !*floating {
                    let panel_id = *panel_id;
                    let title = title.clone();
                    self.dragging_panel_title = Some((id, panel_id));
                    self.hover_region = None;
                    self.dock_tear_title = Some(title);
                    self.dock_tear_press_cursor = Some(self.cursor);
                    self.seed_dock_tear_from_panel(panel_id);
                }
            }
            WidgetKind::Container { .. } | WidgetKind::Label { .. } | WidgetKind::Viewport { .. } => {}
        }
    }

    fn handle_floating_mouse_press(&mut self, window_id: WindowId) {
        let Some(floater) = self.floating.get(&window_id) else { return };
        let cursor = floater.cursor;
        let region_id = floater.region_id;
        // Edge/corner resize takes priority over title-bar move (top few points are resize),
        // except on the title-bar ×.
        if let Some(edge) = Self::floating_resize_edge_at_cursor(floater) {
            let scale = floater.scale_factor.max(0.01);
            let Ok(inner) = floater.window.inner_position() else { return };
            let Ok(outer) = floater.window.outer_position() else { return };
            let cursor_phys =
                LogicalPosition::new(cursor.0 as f64, cursor.1 as f64).to_physical::<f64>(scale);
            self.dragging_floating_resize = Some(FloatingResizeDrag {
                window_id,
                edge,
                start_outer: outer,
                start_inner: PhysicalSize::new(floater.physical_width, floater.physical_height),
                start_cursor_screen: (inner.x as f64 + cursor_phys.x, inner.y as f64 + cursor_phys.y),
            });
            return;
        }
        let Some(id) = floater.widget_tree.hit_test(cursor.0, cursor.1) else { return };
        let Some(kind) = floater.widget_tree.kind(id) else { return };
        match kind {
            WidgetKind::Button { on_click, .. } => {
                if let Some(callback) = on_click.clone() {
                    callback();
                }
            }
            WidgetKind::Slider { .. } => {
                if let Some(floater) = self.floating.get_mut(&window_id) {
                    floater.dragging_slider = Some(id);
                }
                self.update_floating_dragged_slider(window_id);
            }
            WidgetKind::Splitter { .. } => {
                if let Some(floater) = self.floating.get_mut(&window_id) {
                    floater.dragging_splitter = Some(id);
                }
            }
            WidgetKind::TabBar { panel_ids, titles, on_close, .. } => {
                if let Some(index) = Self::floating_tab_bar_clicked_index(floater, id) {
                    if let Some(rect) = floater.widget_tree.absolute_rect(id) {
                        let n = titles.len().max(1) as f32;
                        let segment = fastgui_core::widget::Rect {
                            x: rect.x + index as f32 * (rect.width / n),
                            y: rect.y,
                            width: rect.width / n,
                            height: rect.height,
                        };
                        if on_close.get(index).is_some_and(|c| c.is_some())
                            && fastgui_core::widget::close_button_rect(segment).contains(cursor.0, cursor.1)
                        {
                            if let Some(callback) = on_close[index].clone() {
                                if let Some(&panel_id) = panel_ids.get(index) {
                                    callback(panel_id);
                                }
                            }
                            return;
                        }
                    }
                }
                self.handle_floating_tab_click(window_id, id);
            }
            WidgetKind::PanelTitleBar { panel_id, floating, on_drop, on_close, .. } => {
                if let Some(rect) = floater.widget_tree.absolute_rect(id) {
                    if on_close.is_some()
                        && fastgui_core::widget::close_button_rect(rect).contains(cursor.0, cursor.1)
                    {
                        if let Some(callback) = on_close.clone() {
                            callback(*panel_id);
                        }
                        return;
                    }
                }
                let panel_id = *panel_id;
                let floating = *floating;
                let can_redock = on_drop.is_some();
                if floating {
                    // Grab offset in this window's logical cursor space so `set_outer_position`
                    // keeps the cursor on the title bar while the OS window follows.
                    self.dragging_floating_panel = Some((window_id, cursor));
                    if can_redock {
                        self.dragging_panel_title = Some((id, panel_id));
                        self.dragging_from_floating = Some(region_id);
                        self.hover_region = None;
                    }
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

    fn floating_tab_bar_clicked_index(floater: &FloatingWindow<B>, bar_id: WidgetId) -> Option<usize> {
        let rect = floater.widget_tree.absolute_rect(bar_id)?;
        let Some(WidgetKind::TabBar { titles, .. }) = floater.widget_tree.kind(bar_id) else {
            return None;
        };
        if titles.is_empty() || rect.width <= 0.0 {
            return None;
        }
        let segment_width = rect.width / titles.len() as f32;
        Some((((floater.cursor.0 - rect.x) / segment_width) as usize).min(titles.len() - 1))
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

    fn handle_floating_tab_click(&mut self, window_id: WindowId, bar_id: WidgetId) {
        let Some(floater) = self.floating.get_mut(&window_id) else { return };
        let Some(clicked) = Self::floating_tab_bar_clicked_index(floater, bar_id) else { return };
        let Some(WidgetKind::TabBar { active, content_ids, .. }) = floater.widget_tree.kind(bar_id) else {
            return;
        };
        if clicked == *active {
            return;
        }
        let content_ids = content_ids.clone();

        floater.widget_tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::TabBar { active, .. } = kind {
                *active = clicked;
            }
        });
        for (index, &content_id) in content_ids.iter().enumerate() {
            floater.widget_tree.set_display(content_id, index == clicked);
        }

        if let Some(WidgetKind::TabBar { on_select: Some(callback), .. }) = floater.widget_tree.kind(bar_id)
        {
            callback(clicked);
        }
    }

    fn update_panel_drag_hover(&mut self) {
        let Some((_, dragged_region_id)) = self.dragging_panel_title else { return };

        let (width, height) = (self.width as f32, self.height as f32);
        // A floating OS window maps its screen cursor into main-window space. Far outside,
        // e.g. `cursor.x = width + 400`, still satisfies `>= width - OUTER_EDGE_MARGIN` and
        // would light up a whole-window edge drop — clear the hover unless the pointer is
        // actually over (or just beside) the main window.
        let over_main = self.cursor.0 >= -OUTER_EDGE_MARGIN
            && self.cursor.1 >= -OUTER_EDGE_MARGIN
            && self.cursor.0 <= width + OUTER_EDGE_MARGIN
            && self.cursor.1 <= height + OUTER_EDGE_MARGIN;
        if !over_main {
            self.hover_region = None;
            return;
        }

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

    fn seed_dock_tear_from_panel(&mut self, panel_id: u64) {
        let Some(rect) = self.widget_tree.find_region_rect(panel_id) else {
            self.dock_tear_grab = None;
            self.dock_tear_size = None;
            return;
        };
        self.dock_tear_grab = Some((self.cursor.0 - rect.x, self.cursor.1 - rect.y));
        self.dock_tear_size = Some((rect.width.max(MIN_FLOAT_WIDTH), rect.height.max(MIN_FLOAT_HEIGHT)));
    }

    fn seed_dock_tear_from_tab(&mut self, bar_id: WidgetId, index: usize, content_ids: &[WidgetId]) {
        let Some(bar_rect) = self.widget_tree.absolute_rect(bar_id) else {
            self.dock_tear_grab = None;
            self.dock_tear_size = None;
            return;
        };
        let content_h = content_ids
            .get(index)
            .and_then(|&cid| self.widget_tree.absolute_rect(cid))
            .map(|r| r.height)
            .unwrap_or(180.0);
        self.dock_tear_grab = Some((self.cursor.0 - bar_rect.x, self.cursor.1 - bar_rect.y));
        self.dock_tear_size = Some((
            bar_rect.width.max(MIN_FLOAT_WIDTH),
            (bar_rect.height + content_h).max(MIN_FLOAT_HEIGHT),
        ));
    }

    fn cursor_over_main_window(&self) -> bool {
        let (width, height) = (self.width as f32, self.height as f32);
        self.cursor.0 >= -OUTER_EDGE_MARGIN
            && self.cursor.1 >= -OUTER_EDGE_MARGIN
            && self.cursor.0 <= width + OUTER_EDGE_MARGIN
            && self.cursor.1 <= height + OUTER_EDGE_MARGIN
    }

    fn panel_drop_callback(
        &self,
        bar_id: WidgetId,
        dragged_region_id: u64,
        from_floating: Option<u64>,
    ) -> Option<fastgui_core::widget::PanelDropCallback> {
        if let Some(region_id) = from_floating {
            let window_id = *self.floating_by_region.get(&region_id)?;
            let floater = self.floating.get(&window_id)?;
            return match floater.widget_tree.kind(bar_id) {
                Some(WidgetKind::PanelTitleBar { on_drop: Some(callback), .. }) => Some(callback.clone()),
                Some(WidgetKind::TabBar { panel_ids, on_drop, .. }) => panel_ids
                    .iter()
                    .position(|&id| id == dragged_region_id)
                    .and_then(|index| on_drop[index].clone()),
                _ => None,
            };
        }
        match self.widget_tree.kind(bar_id) {
            Some(WidgetKind::PanelTitleBar { on_drop: Some(callback), .. }) => Some(callback.clone()),
            Some(WidgetKind::TabBar { panel_ids, on_drop, .. }) => panel_ids
                .iter()
                .position(|&id| id == dragged_region_id)
                .and_then(|index| on_drop[index].clone()),
            _ => None,
        }
    }

    fn destroy_tear_ghost(&mut self) {
        self.tear_ghost = None;
        self.dock_tear_title = None;
        self.dock_tear_press_cursor = None;
    }

    fn spawn_tear_ghost(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RunError<B::Error>> {
        let Some(main) = &self.window else { return Ok(()) };
        let (fw, fh) = self.dock_tear_size.unwrap_or((320.0, 180.0));
        let (gx, gy) = self.dock_tear_grab.unwrap_or((fw * 0.5, 14.0));
        let title = self.dock_tear_title.clone().unwrap_or_else(|| "Panel".into());
        let scale = main.scale_factor().max(0.01);
        let position = match main.inner_position() {
            Ok(inner) => {
                let logical = inner.to_logical::<f64>(scale);
                LogicalPosition::new(
                    logical.x + f64::from(self.cursor.0 - gx),
                    logical.y + f64::from(self.cursor.1 - gy),
                )
            }
            Err(_) => LogicalPosition::new(f64::from(self.cursor.0 - gx), f64::from(self.cursor.1 - gy)),
        };

        let attributes = Window::default_attributes()
            .with_title(title.clone())
            .with_inner_size(LogicalSize::new(f64::from(fw), f64::from(fh)))
            .with_position(position)
            .with_decorations(false)
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop);
        let window = event_loop.create_window(attributes)?;
        let _ = window.set_cursor_hittest(false);
        let scale_factor = window.scale_factor();
        let physical = window.inner_size();
        let logical = physical.to_logical::<f64>(scale_factor.max(0.01));
        let mut renderer = B::new(&window, physical.width, physical.height).map_err(RunError::Renderer)?;
        let mut widget_tree = build_tear_ghost_tree(&title);
        let mut chrome = ChromeRenderer::new();
        widget_tree.compute_layout(logical.width as f32, logical.height as f32);
        let frame = chrome.rasterize(
            &widget_tree,
            physical.width,
            physical.height,
            None,
            scale_factor as f32,
        );
        widget_tree.clear_dirty();
        renderer.set_chrome_frame(frame).map_err(RunError::Renderer)?;
        renderer
            .render_frame(TEAR_GHOST_CLEAR, true, &[])
            .map_err(RunError::Renderer)?;
        window.request_redraw();

        // Chrome is baked into the renderer once; the ghost only needs to follow the cursor.
        drop((widget_tree, chrome, logical, scale_factor));
        self.tear_ghost = Some(TearGhost {
            window,
            renderer,
            grab_offset: (gx, gy),
        });
        Ok(())
    }

    /// After a short drag threshold, show a panel-shaped ghost that tracks the cursor while the
    /// real docked panel stays put until mouse-up (Qt non-opaque / "preview" undock style).
    fn update_tear_ghost(&mut self, event_loop: &ActiveEventLoop) {
        if self.dragging_from_floating.is_some() || self.dragging_panel_title.is_none() {
            return;
        }
        if self.tear_ghost.is_none() {
            let Some(press) = self.dock_tear_press_cursor else { return };
            let dx = self.cursor.0 - press.0;
            let dy = self.cursor.1 - press.1;
            if dx * dx + dy * dy < TEAR_GHOST_THRESHOLD * TEAR_GHOST_THRESHOLD {
                return;
            }
            // A new drag replaces any ghost still covering a previous tear-off's floater.
            self.tear_handoff = None;
            if let Err(err) = self.spawn_tear_ghost(event_loop) {
                self.error = Some(err);
                event_loop.exit();
                return;
            }
        }
        let Some(main) = &self.window else { return };
        let Some(ghost) = &self.tear_ghost else { return };
        let scale = main.scale_factor().max(0.01);
        let Ok(inner) = main.inner_position() else { return };
        let logical = inner.to_logical::<f64>(scale);
        let (gx, gy) = ghost.grab_offset;
        ghost.window.set_outer_position(LogicalPosition::new(
            logical.x + f64::from(self.cursor.0 - gx),
            logical.y + f64::from(self.cursor.1 - gy),
        ));
    }

    fn handle_panel_drop(&mut self) {
        // Held locally: every path drops it here except a tear-off, which hands it to
        // `tear_handoff` to cover the new floater until its first frame.
        let ghost = self.tear_ghost.take();
        self.destroy_tear_ghost();
        let Some((bar_id, dragged_region_id)) = self.dragging_panel_title.take() else {
            self.dock_tear_grab = None;
            self.dock_tear_size = None;
            return;
        };
        let hover = self.hover_region.take();
        let from_floating = self.dragging_from_floating.take();
        let tear_grab = self.dock_tear_grab.take();
        let tear_size = self.dock_tear_size.take();

        let Some(callback) = self.panel_drop_callback(bar_id, dragged_region_id, from_floating) else {
            return;
        };

        if let Some((target_region_id, _, zone)) = hover {
            callback(dragged_region_id, target_region_id, zone, None);
            return;
        }

        // Released with no dock hover: tear a *docked* panel out into a floating OS window when
        // the cursor is outside the main window. Floating→floating (already floating, no hover)
        // is a no-op — leave it where it was moved.
        if from_floating.is_some() || self.cursor_over_main_window() {
            return;
        }
        let (fw, fh) = tear_size.unwrap_or((320.0, 180.0));
        let (gx, gy) = tear_grab.unwrap_or((fw * 0.5, 14.0));
        let fx = self.cursor.0 - gx;
        let fy = self.cursor.1 - gy;
        callback(
            dragged_region_id,
            ROOT_REGION_ID,
            fastgui_core::widget::DropZone::Float,
            Some((fx, fy, fw, fh)),
        );
        if ghost.is_some() {
            self.tear_ghost = ghost;
            self.tear_handoff = Some(dragged_region_id);
        }
    }

    /// Apply an in-progress edge/corner resize using screen-space deltas from press.
    fn update_floating_resize(&mut self) {
        let Some(drag) = self.dragging_floating_resize.as_ref() else { return };
        let window_id = drag.window_id;
        let edge = drag.edge;
        let start_outer = drag.start_outer;
        let start_inner = drag.start_inner;
        let start_cursor = drag.start_cursor_screen;

        let Some(floater) = self.floating.get(&window_id) else { return };
        let scale = floater.scale_factor.max(0.01);
        let Ok(inner) = floater.window.inner_position() else { return };
        let cursor_phys =
            LogicalPosition::new(floater.cursor.0 as f64, floater.cursor.1 as f64).to_physical::<f64>(scale);
        let screen_x = inner.x as f64 + cursor_phys.x;
        let screen_y = inner.y as f64 + cursor_phys.y;
        let dx = screen_x - start_cursor.0;
        let dy = screen_y - start_cursor.1;

        let min_w = (MIN_FLOAT_WIDTH as f64 * scale).max(1.0);
        let min_h = (MIN_FLOAT_HEIGHT as f64 * scale).max(1.0);
        let mut width = start_inner.width as f64;
        let mut height = start_inner.height as f64;
        let mut outer_x = start_outer.x as f64;
        let mut outer_y = start_outer.y as f64;

        match edge {
            ResizeEdge::Right | ResizeEdge::TopRight | ResizeEdge::BottomRight => {
                width = (start_inner.width as f64 + dx).max(min_w);
            }
            ResizeEdge::Left | ResizeEdge::TopLeft | ResizeEdge::BottomLeft => {
                let new_w = (start_inner.width as f64 - dx).max(min_w);
                outer_x = start_outer.x as f64 + (start_inner.width as f64 - new_w);
                width = new_w;
            }
            ResizeEdge::Top | ResizeEdge::Bottom => {}
        }
        match edge {
            ResizeEdge::Bottom | ResizeEdge::BottomLeft | ResizeEdge::BottomRight => {
                height = (start_inner.height as f64 + dy).max(min_h);
            }
            ResizeEdge::Top | ResizeEdge::TopLeft | ResizeEdge::TopRight => {
                let new_h = (start_inner.height as f64 - dy).max(min_h);
                outer_y = start_outer.y as f64 + (start_inner.height as f64 - new_h);
                height = new_h;
            }
            ResizeEdge::Left | ResizeEdge::Right => {}
        }

        floater.window.set_outer_position(PhysicalPosition::new(outer_x as i32, outer_y as i32));
        let _ = floater.window.request_inner_size(PhysicalSize::new(
            width.round().max(1.0) as u32,
            height.round().max(1.0) as u32,
        ));
    }

    /// Move the floating OS window so its grab point stays under the cursor, then map that
    /// screen point into the main window's logical space for dock drop-zone hover.
    fn update_dragged_floating_panel(&mut self) {
        let Some((window_id, (offset_x, offset_y))) = self.dragging_floating_panel else { return };
        let (scale, cursor, inner, outer) = {
            let Some(floater) = self.floating.get(&window_id) else { return };
            let scale = floater.scale_factor.max(0.01);
            let Ok(inner) = floater.window.inner_position() else { return };
            let Ok(outer) = floater.window.outer_position() else { return };
            (scale, floater.cursor, inner, outer)
        };
        let cursor_phys = LogicalPosition::new(cursor.0 as f64, cursor.1 as f64).to_physical::<f64>(scale);
        let offset_phys = LogicalPosition::new(offset_x as f64, offset_y as f64).to_physical::<f64>(scale);
        let screen_x = inner.x as f64 + cursor_phys.x;
        let screen_y = inner.y as f64 + cursor_phys.y;

        // Drop hover must use the pre-move screen cursor; after `set_outer_position` the
        // window-relative cursor is stale until the next `CursorMoved`.
        if self.dragging_from_floating.is_some() {
            self.sync_main_cursor_from_screen(screen_x, screen_y);
            self.update_panel_drag_hover();
        }

        let Some(floater) = self.floating.get(&window_id) else { return };
        let new_inner_x = (screen_x - offset_phys.x) as i32;
        let new_inner_y = (screen_y - offset_phys.y) as i32;
        // Undecorated windows usually have outer == inner; keep the delta anyway.
        floater.window.set_outer_position(PhysicalPosition::new(
            new_inner_x + (outer.x - inner.x),
            new_inner_y + (outer.y - inner.y),
        ));
    }

    /// Map a desktop physical point into the main window's logical client space.
    fn sync_main_cursor_from_screen(&mut self, screen_x: f64, screen_y: f64) {
        let Some(main) = &self.window else { return };
        let Ok(main_inner) = main.inner_position() else { return };
        let main_scale = self.scale_factor.max(0.01);
        self.cursor = (
            ((screen_x - main_inner.x as f64) / main_scale) as f32,
            ((screen_y - main_inner.y as f64) / main_scale) as f32,
        );
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

    fn update_floating_dragged_splitter(&mut self, window_id: WindowId) {
        let Some(floater) = self.floating.get_mut(&window_id) else { return };
        let Some(bar_id) = floater.dragging_splitter else { return };
        let Some(WidgetKind::Splitter { direction, first, second, .. }) = floater.widget_tree.kind(bar_id)
        else {
            return;
        };
        let (direction, first, second) = (*direction, *first, *second);
        let Some(first_rect) = floater.widget_tree.absolute_rect(first) else { return };
        let Some(second_rect) = floater.widget_tree.absolute_rect(second) else { return };
        let cursor = floater.cursor;

        let (cursor_pos, start, end) = match direction {
            fastgui_core::widget::SplitDirection::Row => {
                (cursor.0, first_rect.x, second_rect.x + second_rect.width)
            }
            fastgui_core::widget::SplitDirection::Column => {
                (cursor.1, first_rect.y, second_rect.y + second_rect.height)
            }
        };
        let span = end - start;
        if span <= 0.0 {
            return;
        }
        const MIN_FRACTION: f32 = 0.05;
        let ratio = ((cursor_pos - start) / span).clamp(MIN_FRACTION, 1.0 - MIN_FRACTION);

        floater.widget_tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::Splitter { ratio: current, .. } = kind {
                *current = ratio;
            }
        });
        const GROW_SCALE: f32 = 1000.0;
        floater.widget_tree.set_flex_grow(first, ratio * GROW_SCALE);
        floater.widget_tree.set_flex_grow(second, (1.0 - ratio) * GROW_SCALE);
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

    fn update_floating_dragged_slider(&mut self, window_id: WindowId) {
        let Some(floater) = self.floating.get_mut(&window_id) else { return };
        let Some(id) = floater.dragging_slider else { return };
        let Some(rect) = floater.widget_tree.absolute_rect(id) else { return };
        let fraction = if rect.width > 0.0 {
            ((floater.cursor.0 - rect.x) / rect.width).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut changed_value = None;
        floater.widget_tree.mutate_kind(id, |kind| {
            if let WidgetKind::Slider { value, min, max, .. } = kind {
                *value = *min + fraction * (*max - *min);
                changed_value = Some(*value);
            }
        });

        if let (Some(value), Some(WidgetKind::Slider { on_change: Some(callback), .. })) =
            (changed_value, floater.widget_tree.kind(id))
        {
            callback(value);
        }
    }

    fn update_cursor_icon(&mut self) {
        let Some(window) = &self.window else { return };
        if self.dragging_panel_title.is_some() || self.dragging_floating_panel.is_some() {
            window.set_cursor(GRABBING_CURSOR);
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
                Some(GRAB_CURSOR)
            }
            _ => None,
        };
        window.set_cursor(icon.unwrap_or(winit::window::CursorIcon::Default));
    }

    fn update_floating_cursor_icon(&mut self, window_id: WindowId) {
        let resizing = self.dragging_floating_resize.as_ref().is_some_and(|d| d.window_id == window_id);
        let dragging = self.dragging_floating_panel.is_some_and(|(id, _)| id == window_id)
            || self
                .dragging_from_floating
                .is_some_and(|rid| self.floating_by_region.get(&rid) == Some(&window_id));
        let Some(floater) = self.floating.get(&window_id) else { return };
        if resizing {
            if let Some(drag) = &self.dragging_floating_resize {
                floater.window.set_cursor(resize_edge_cursor(drag.edge));
            }
            return;
        }
        if dragging {
            floater.window.set_cursor(GRABBING_CURSOR);
            return;
        }
        if let Some(edge) = Self::floating_resize_edge_at_cursor(floater) {
            floater.window.set_cursor(resize_edge_cursor(edge));
            return;
        }
        let hovered = floater
            .dragging_splitter
            .or_else(|| floater.widget_tree.hit_test(floater.cursor.0, floater.cursor.1));
        let icon = match hovered.and_then(|id| floater.widget_tree.kind(id)) {
            Some(WidgetKind::Splitter { direction, .. }) => Some(match direction {
                fastgui_core::widget::SplitDirection::Row => winit::window::CursorIcon::ColResize,
                fastgui_core::widget::SplitDirection::Column => winit::window::CursorIcon::RowResize,
            }),
            Some(WidgetKind::PanelTitleBar { on_drop, floating, .. }) if on_drop.is_some() || *floating => {
                Some(GRAB_CURSOR)
            }
            _ => None,
        };
        floater
            .window
            .set_cursor(icon.unwrap_or(winit::window::CursorIcon::Default));
    }

    fn apply_physical_size(&mut self, physical: PhysicalSize<u32>) {
        self.physical_width = physical.width;
        self.physical_height = physical.height;
        if let Some(window) = &self.window {
            self.scale_factor = window.scale_factor();
        }
        let logical = physical.to_logical::<f64>(self.scale_factor.max(0.01));
        self.width = logical.width.round().max(0.0) as u32;
        self.height = logical.height.round().max(0.0) as u32;
    }

    fn apply_floating_physical_size(floater: &mut FloatingWindow<B>, physical: PhysicalSize<u32>) {
        floater.physical_width = physical.width;
        floater.physical_height = physical.height;
        floater.scale_factor = floater.window.scale_factor();
        let logical = physical.to_logical::<f64>(floater.scale_factor.max(0.01));
        floater.width = logical.width.round().max(0.0) as u32;
        floater.height = logical.height.round().max(0.0) as u32;
    }

    /// Drain queued commands and render one frame immediately (main + every floater). Used by
    /// drag/resize paths, where a floater drag also has to repaint the main window's drop
    /// highlight. `RedrawRequested` uses `render_window` instead.
    fn render_now(&mut self, event_loop: &ActiveEventLoop) {
        self.drain_commands(event_loop);
        if !self.render_main(event_loop) {
            return;
        }
        let floating_ids: Vec<WindowId> = self.floating.keys().copied().collect();
        for window_id in floating_ids {
            if !self.render_floater(event_loop, window_id) {
                return;
            }
        }
        if let Some(ghost) = &mut self.tear_ghost {
            if let Err(err) = ghost.renderer.render_frame(TEAR_GHOST_CLEAR, true, &[]) {
                self.error = Some(RunError::Renderer(err));
                event_loop.exit();
            }
        }
    }

    /// Drain queued commands and render only `window_id` — what that window's own
    /// `RedrawRequested` does. A wake asks every window to redraw, so rendering all of them
    /// from each redraw would cost (floaters + 1)² frames per wake instead of one per window.
    fn render_window(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId) {
        self.drain_commands(event_loop);
        if self.window.as_ref().is_some_and(|w| w.id() == window_id) {
            self.render_main(event_loop);
        } else {
            self.render_floater(event_loop, window_id);
        }
    }

    /// Render the main window. Returns `false` (after recording the error and exiting) on
    /// failure.
    fn render_main(&mut self, event_loop: &ActiveEventLoop) -> bool {
        // Debounced policy: catch the surface up to the latest size only once a resize has
        // settled — see `RESIZE_DEBOUNCE`. Runs on every main-window frame so the first frame
        // after the debounce window is always correctly sized.
        if matches!(self.resize_policy, MainResizePolicy::Debounced) {
            let settled = self.last_resize_event.elapsed() >= RESIZE_DEBOUNCE;
            if settled && self.last_applied_physical != (self.physical_width, self.physical_height) {
                if let Some(renderer) = &mut self.renderer {
                    if let Err(err) = renderer.resize(self.physical_width, self.physical_height) {
                        self.error = Some(RunError::Renderer(err));
                        event_loop.exit();
                        return false;
                    }
                }
                self.last_applied_physical = (self.physical_width, self.physical_height);
            }
        }

        if let Err(err) = self.upload_active_content() {
            self.error = Some(RunError::Renderer(err));
            event_loop.exit();
            return false;
        }
        let draws = self.viewport_draws();
        let draw_chrome = self.has_widget_content;
        let clear_color = self.clear_color;
        if let Some(renderer) = &mut self.renderer {
            if let Err(err) = renderer.render_frame(clear_color, draw_chrome, &draws) {
                self.error = Some(RunError::Renderer(err));
                event_loop.exit();
                return false;
            }
        }
        true
    }

    /// Render one floating window. Returns `false` (after recording the error and exiting) on
    /// failure; a floater that's already gone is not a failure.
    fn render_floater(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId) -> bool {
        let clear_color = self.clear_color;
        let Some(floater) = self.floating.get_mut(&window_id) else { return true };
        if let Err(err) = Self::upload_floating_content(floater) {
            self.error = Some(RunError::Renderer(err));
            event_loop.exit();
            return false;
        }
        let draws = Self::floating_viewport_draws(floater);
        let draw_chrome = floater.has_widget_content;
        if let Err(err) = floater.renderer.render_frame(clear_color, draw_chrome, &draws) {
            self.error = Some(RunError::Renderer(err));
            event_loop.exit();
            return false;
        }
        if self.tear_handoff == Some(floater.region_id) {
            self.tear_handoff = None;
            self.tear_ghost = None;
        }
        true
    }

    fn schedule_control_flow(&self, event_loop: &ActiveEventLoop) {
        if matches!(self.resize_policy, MainResizePolicy::Debounced)
            && self.last_applied_physical != (self.physical_width, self.physical_height)
        {
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

impl<B: SurfaceBackend> ApplicationHandler for App<B> {
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

        self.scale_factor = window.scale_factor();
        self.apply_physical_size(window.inner_size());

        match B::new(&window, self.physical_width, self.physical_height) {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.last_applied_physical = (self.physical_width, self.physical_height);
                window.request_redraw();
                self.window = Some(window);
            }
            Err(err) => {
                self.error = Some(RunError::Renderer(err));
                event_loop.exit();
                return;
            }
        }
        let pending = std::mem::take(&mut self.pending_floaters);
        for (region_id, title, x, y, width, height, build) in pending {
            if let Err(err) =
                self.add_floating_panel(event_loop, region_id, title, x, y, width, height, build)
            {
                self.error = Some(err);
                event_loop.exit();
                return;
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let is_main = self.window.as_ref().is_some_and(|w| w.id() == window_id);
        if !is_main {
            if self.floating.contains_key(&window_id) {
                self.handle_floating_window_event(event_loop, window_id, event);
            } else if self.tear_ghost.as_ref().is_some_and(|g| g.window.id() == window_id) {
                if matches!(event, WindowEvent::RedrawRequested) {
                    if let Some(ghost) = &mut self.tear_ghost {
                        if let Err(err) = ghost.renderer.render_frame(TEAR_GHOST_CLEAR, true, &[]) {
                            self.error = Some(RunError::Renderer(err));
                            event_loop.exit();
                        }
                    }
                }
            }
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
            }
            WindowEvent::Resized(size) => {
                self.apply_physical_size(size);
                match self.resize_policy {
                    MainResizePolicy::Immediate => {
                        if let Some(renderer) = &mut self.renderer {
                            if let Err(err) =
                                renderer.resize(self.physical_width, self.physical_height)
                            {
                                self.error = Some(RunError::Renderer(err));
                                event_loop.exit();
                                return;
                            }
                        }
                        self.last_applied_physical = (self.physical_width, self.physical_height);
                        self.render_now(event_loop);
                    }
                    MainResizePolicy::Debounced => {
                        self.last_resize_event = Instant::now();
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let logical = position.to_logical::<f64>(self.scale_factor.max(0.01));
                self.cursor = (logical.x as f32, logical.y as f32);
                let dragging = self.dragging_slider.is_some()
                    || self.dragging_splitter.is_some()
                    || self.dragging_panel_title.is_some();
                if self.dragging_slider.is_some() {
                    self.update_dragged_slider();
                }
                if self.dragging_splitter.is_some() {
                    self.update_dragged_splitter();
                }
                if self.dragging_panel_title.is_some() && self.dragging_from_floating.is_none() {
                    self.update_panel_drag_hover();
                    self.update_tear_ghost(event_loop);
                }
                self.update_cursor_icon();
                if dragging && self.last_drag_render.elapsed() >= DRAG_RENDER_INTERVAL {
                    self.render_now(event_loop);
                    self.last_drag_render = Instant::now();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => self.handle_mouse_press(),
                ElementState::Released => {
                    let was_dragging = self.dragging_slider.is_some()
                        || self.dragging_splitter.is_some()
                        || self.dragging_panel_title.is_some();
                    self.dragging_slider = None;
                    self.dragging_splitter = None;
                    // Floater OS-window drag is owned by the floater's mouse-up path.
                    if self.dragging_from_floating.is_none() {
                        self.handle_panel_drop();
                    }
                    self.update_cursor_icon();
                    if was_dragging {
                        self.render_now(event_loop);
                    }
                }
            },
            WindowEvent::RedrawRequested => {
                self.render_window(event_loop, window_id);
                self.schedule_control_flow(event_loop);
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
        for floater in self.floating.values() {
            floater.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.schedule_control_flow(event_loop);
    }
}

impl<B: SurfaceBackend> App<B> {
    fn handle_floating_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            // Alt+F4 etc. Go through the panel's close handler, same as its ×, so Python drops
            // it from `floating_panels` too — destroying just the OS window here would leave
            // the panel listed as floating with nothing on screen to get it back. Every floater
            // has one (`fastgui-py`'s `bind_panel_to_dock_handlers` falls back to the window's
            // own `_take_floating_panel`), so the `None` case is only a safety net.
            WindowEvent::CloseRequested => {
                if let Some(callback) = self.floating_close_callback(window_id) {
                    let region_id = self.floating[&window_id].region_id;
                    callback(region_id);
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(floater) = self.floating.get_mut(&window_id) {
                    floater.scale_factor = scale_factor;
                }
            }
            WindowEvent::Resized(size) => {
                let resize_result = {
                    let Some(floater) = self.floating.get_mut(&window_id) else { return };
                    // Windows sends a same-size `Resized` when a new floater is first shown.
                    // Recreating the swapchain for it costs a ~2s DWM stall (see
                    // `RESIZE_DEBOUNCE`) with the window left blank, so skip no-op resizes.
                    let unchanged = (size.width, size.height) == (floater.physical_width, floater.physical_height);
                    Self::apply_floating_physical_size(floater, size);
                    if unchanged {
                        Ok(())
                    } else {
                        floater
                            .renderer
                            .resize(floater.physical_width, floater.physical_height)
                    }
                };
                if let Err(err) = resize_result {
                    self.error = Some(RunError::Renderer(err));
                    event_loop.exit();
                    return;
                }
                self.render_now(event_loop);
            }
            WindowEvent::CursorMoved { position, .. } => {
                {
                    let Some(floater) = self.floating.get_mut(&window_id) else { return };
                    let logical = position.to_logical::<f64>(floater.scale_factor.max(0.01));
                    floater.cursor = (logical.x as f32, logical.y as f32);
                }
                let dragging_widget = self.floating.get(&window_id).is_some_and(|f| {
                    f.dragging_slider.is_some() || f.dragging_splitter.is_some()
                });
                let dragging_window = self.dragging_floating_panel.is_some_and(|(id, _)| id == window_id);
                let resizing =
                    self.dragging_floating_resize.as_ref().is_some_and(|d| d.window_id == window_id);
                if self.floating.get(&window_id).is_some_and(|f| f.dragging_slider.is_some()) {
                    self.update_floating_dragged_slider(window_id);
                }
                if self.floating.get(&window_id).is_some_and(|f| f.dragging_splitter.is_some()) {
                    self.update_floating_dragged_splitter(window_id);
                }
                if resizing {
                    self.update_floating_resize();
                } else if dragging_window {
                    self.update_dragged_floating_panel();
                }
                self.update_floating_cursor_icon(window_id);
                if (dragging_widget || dragging_window || resizing)
                    && self.last_drag_render.elapsed() >= DRAG_RENDER_INTERVAL
                {
                    self.render_now(event_loop);
                    self.last_drag_render = Instant::now();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => self.handle_floating_mouse_press(window_id),
                ElementState::Released => {
                    let was_dragging = self.floating.get(&window_id).is_some_and(|f| {
                        f.dragging_slider.is_some() || f.dragging_splitter.is_some()
                    }) || self.dragging_floating_panel.is_some_and(|(id, _)| id == window_id)
                        || self.dragging_floating_resize.as_ref().is_some_and(|d| d.window_id == window_id)
                        || self.dragging_from_floating.is_some();
                    if let Some(floater) = self.floating.get_mut(&window_id) {
                        floater.dragging_slider = None;
                        floater.dragging_splitter = None;
                    }
                    self.dragging_floating_panel = None;
                    self.dragging_floating_resize = None;
                    self.handle_panel_drop();
                    self.update_floating_cursor_icon(window_id);
                    if was_dragging {
                        self.render_now(event_loop);
                    }
                }
            },
            WindowEvent::RedrawRequested => {
                self.render_window(event_loop, window_id);
                self.schedule_control_flow(event_loop);
            }
            _ => {}
        }
    }
}

/// Open a window titled `title` at logical `width`×`height`, initialize backend `B` against it,
/// and block the calling thread until the window is closed.
pub fn run<B: SurfaceBackend>(
    title: &str,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
    handles: RenderThreadHandles,
    resize_policy: MainResizePolicy,
) -> Result<(), RunError<B::Error>> {
    let event_loop = EventLoop::new()?;
    handles.waker.bind(event_loop.create_proxy());
    let mut app = App::<B> {
        title: title.to_owned(),
        width,
        height,
        physical_width: width,
        physical_height: height,
        scale_factor: 1.0,
        clear_color,
        handles,
        resize_policy,
        widget_tree: WidgetTree::new(),
        chrome: ChromeRenderer::new(),
        last_chrome_size: None,
        last_raster_scale: 0.0,
        has_widget_content: false,
        cursor: (0.0, 0.0),
        dragging_slider: None,
        dragging_splitter: None,
        dragging_panel_title: None,
        hover_region: None,
        dragging_floating_panel: None,
        dragging_floating_resize: None,
        dragging_from_floating: None,
        dock_tear_grab: None,
        dock_tear_size: None,
        dock_tear_title: None,
        dock_tear_press_cursor: None,
        tear_ghost: None,
        tear_handoff: None,
        last_drag_render: Instant::now(),
        last_applied_physical: (width, height),
        last_resize_event: Instant::now(),
        window: None,
        renderer: None,
        floating: HashMap::new(),
        floating_by_region: HashMap::new(),
        pending_floaters: Vec::new(),
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}
