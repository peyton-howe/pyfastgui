use std::time::Duration;

/// Cap on how often drag `CursorMoved` renders synchronously.
pub const DRAG_RENDER_INTERVAL: Duration = Duration::from_millis(1000 / 60);

/// Reserved `region_id` meaning "the whole DockArea" (window outer-edge drop).
pub const ROOT_REGION_ID: u64 = 0;

/// How close to the window edge (layout units) to prefer a whole-DockArea split.
pub const OUTER_EDGE_MARGIN: f32 = 24.0;

/// Hit-test margin for floating-window edge/corner resize (layout units).
pub const FLOAT_RESIZE_MARGIN: f32 = 6.0;

/// Minimum floating-window inner size in logical points (Python / `add_floating_panel` space).
pub const MIN_FLOAT_WIDTH: f32 = 160.0;
pub const MIN_FLOAT_HEIGHT: f32 = 100.0;

/// How far a docked-panel drag must move before the tear-off ghost appears (layout units).
pub const TEAR_GHOST_THRESHOLD: f32 = 6.0;

/// Ghost body clear color.
pub const TEAR_GHOST_CLEAR: [f32; 4] = [0.16, 0.20, 0.28, 1.0];

/// Vulkan main-window swapchain recreate debounce (Metal uses immediate resize).
pub const RESIZE_DEBOUNCE: Duration = Duration::from_millis(150);
