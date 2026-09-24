from typing import Any, Callable, Sequence, Union

RGBA = tuple[float, float, float, float]

class CudaSurface:
    """Unverified: see Viewport.create_cuda_surface."""

    @property
    def device_ptr(self) -> int: ...
    @property
    def pitch(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def signal_ready(self) -> None: ...

class Viewport:
    def __init__(self) -> None: ...
    def submit_frame(self, data: Any) -> None:
        """Submit a (H, W, 3|4) uint8 frame. Copies into an owned buffer today; a packed-RGBA
        high-FPS path will want fewer copies later."""
        ...
    def create_cuda_surface(self, width: int, height: int) -> CudaSurface: ...

class Label:
    def __init__(self, text: str, font_size: float = 16.0, color: RGBA = (1.0, 1.0, 1.0, 1.0)) -> None: ...
    def set_text(self, text: str) -> None: ...

class Button:
    def __init__(
        self,
        text: str,
        on_click: Callable[[], None] | None = None,
        font_size: float = 16.0,
        text_color: RGBA = (1.0, 1.0, 1.0, 1.0),
        background: RGBA = (0.25, 0.35, 0.85, 1.0),
    ) -> None: ...
    def set_text(self, text: str) -> None: ...

class Slider:
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 1.0,
        on_change: Callable[[float], None] | None = None,
        track_color: RGBA = (0.3, 0.3, 0.35, 1.0),
        thumb_color: RGBA = (0.4, 0.7, 1.0, 1.0),
    ) -> None: ...
    def set_value(self, value: float) -> None: ...

Widget = Union[Label, Button, Slider, "Box", "Splitter", "Panel", "Tabs", "DockArea", Viewport]

class Box:
    def __init__(
        self,
        children: Sequence[Widget],
        direction: str = "column",
        gap: float = 0.0,
        padding: float = 0.0,
        flex_grow: float = 0.0,
        width: float | None = None,
        height: float | None = None,
        background: RGBA = (0.0, 0.0, 0.0, 0.0),
    ) -> None: ...

class Splitter:
    """A draggable divider between `first` and `second`. `ratio` is `first`'s initial share
    (0..1) of the space along `direction`; dragging the bar resizes both panes live."""

    def __init__(
        self,
        first: Widget,
        second: Widget,
        direction: str = "row",
        ratio: float = 0.5,
        bar_color: RGBA = (0.2, 0.21, 0.24, 1.0),
        thickness: float = 6.0,
    ) -> None: ...

class Panel:
    """A titled container: a fixed-height title bar above `content`, which fills the rest. The
    title bar is draggable — grab it and drop it on another `DockArea` region to move this panel
    there, if `DockArea.add_panel` has bound a rearrange handler to it (`.id`/
    `set_rearrange_handler` are that plumbing; not meant to be used directly)."""

    def __init__(
        self,
        title: str,
        content: Widget,
        title_font_size: float = 14.0,
        title_color: RGBA = (0.92, 0.93, 0.95, 1.0),
        title_background: RGBA = (0.16, 0.17, 0.20, 1.0),
        background: RGBA = (0.12, 0.13, 0.15, 1.0),
        title_height: float = 28.0,
    ) -> None: ...
    @property
    def id(self) -> int:
        """Stable drag-and-drop identity, assigned once at construction. Used internally by
        `DockArea`; only useful directly if you're building your own dock-like container."""
        ...
    def set_rearrange_handler(self, handler: Callable[[int, int, str], None]) -> None:
        """Internal: called by `DockArea.add_panel`. `handler(dragged_id, target_id, zone)`
        fires when this panel's title bar is dragged and dropped on another region — `zone` is
        one of `"center"`, `"left"`, `"right"`, `"top"`, `"bottom"`."""
        ...

class Tabs:
    """Several `Panel`s sharing one region: one combined header strip (each panel's `title`,
    not its own title bar) with only the active tab's content visible. Click a header segment
    to switch, or drag one out onto another `DockArea` region to ungroup it. Also a drop
    target: drop a `Panel` on an edge to split the region, or on the center to add another tab
    to this group."""

    def __init__(
        self,
        panels: Sequence[Panel],
        active: int = 0,
        font_size: float = 14.0,
        text_color: RGBA = (0.92, 0.93, 0.95, 1.0),
        active_color: RGBA = (0.20, 0.22, 0.26, 1.0),
        inactive_color: RGBA = (0.14, 0.15, 0.18, 1.0),
        height: float = 28.0,
        on_select: Callable[[int], None] | None = None,
    ) -> None: ...
    @property
    def id(self) -> int:
        """Stable drag-and-drop identity, assigned once at construction."""
        ...
    @property
    def panels(self) -> Sequence[Panel]:
        """This group's member panels, in tab order."""
        ...
    @property
    def active(self) -> int:
        """This group's currently-active tab index."""
        ...

class DockArea:
    """A split-tree of titled `Panel`s (or `Tabs` groups), built one `add_panel` call at a
    time, with drag-to-rearrange: drag a `Panel`'s title bar (or a tab's own header segment)
    onto another region to move it, onto an existing `Tabs` center to add a tab, or outside
    the main window to tear it out into a floating OS window. Click × on a title bar or tab
    to close. Floating panels can be dropped back into this dock."""

    def __init__(self) -> None: ...
    def add_panel(self, panel: Panel | Tabs, region: str = "center", size: float = 0.25) -> None: ...

class Window:
    def __init__(self, title: str = "fastgui", width: int = 1280, height: int = 720) -> None: ...
    def set_clear_color(self, r: float, g: float, b: float, a: float) -> None: ...
    def set_viewport(self, viewport: Viewport) -> None: ...
    def set_content(self, widget: Widget) -> None: ...
    def add_floating_panel(self, panel: Panel, x: float, y: float, width: float, height: float) -> None:
        """Open `panel` as a real OS window at `(x, y)` relative to this window's inner origin,
        sized `(width, height)`. Draggable anywhere on screen; resize via edge/corner drag; if
        this window's content is a `DockArea`, droppable back onto a docked region to re-dock
        (whether that `set_content` call came before or after this one). Survives later
        `set_content` calls until re-docked or closed. Always closeable via its title-bar ×
        (or the OS close shortcut, e.g. Alt+F4), whatever the window's content is."""
        ...
    @property
    def clear_color(self) -> tuple[float, float, float, float]: ...
    def run(self) -> None: ...
