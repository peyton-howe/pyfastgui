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
    def create_cuda_surface(self, width: int, height: int) -> CudaSurface:
        """Unverified, Windows-only zero-copy CUDA surface. Call after set_viewport(), once
        window.run() has started (from a callback or another thread). Raises RuntimeError
        instead of blocking if run() hasn't started within ~2s, and on macOS/Linux
        (not supported / not implemented yet)."""
        ...

class Label:
    def __init__(self, text: str, font_size: float = 16.0, color: RGBA | None = None) -> None: ...
    def set_text(self, text: str) -> None: ...

class Button:
    def __init__(
        self,
        text: str,
        on_click: Callable[[], None] | None = None,
        font_size: float = 16.0,
        text_color: RGBA | None = None,
        background: RGBA | None = None,
    ) -> None: ...
    def set_text(self, text: str) -> None: ...

class Slider:
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 1.0,
        on_change: Callable[[float], None] | None = None,
        track_color: RGBA | None = None,
        thumb_color: RGBA | None = None,
    ) -> None: ...
    def set_value(self, value: float) -> None: ...

class TextInput:
    """A single-line editable text field. Click or Tab to focus; supports selection, word
    motion, undo/redo, the system clipboard and IME input."""
    def __init__(
        self,
        text: str = "",
        placeholder: str = "",
        on_change: Callable[[str], None] | None = None,
        on_submit: Callable[[str], None] | None = None,
        font_size: float = 16.0,
        width: float | None = None,
        flex_grow: float = 0.0,
        text_color: RGBA | None = None,
        placeholder_color: RGBA | None = None,
        background: RGBA | None = None,
        selection_color: RGBA | None = None,
    ) -> None: ...
    @property
    def text(self) -> str:
        """The current text, including the user's latest edits. Safe from any thread."""
    def set_text(self, text: str) -> None:
        """Replace the text without calling `on_change`. Works before the field is attached."""

class ScrollArea:
    """A scrollable viewport onto `content`. The mouse wheel / trackpad scrolls it (the innermost
    scroll area that can still move takes the scroll), overflowing axes get draggable overlay
    scrollbars, and content outside it is clipped. Tab-focusing a widget scrolls it into view."""
    def __init__(
        self,
        content: Widget,
        flex_grow: float = 1.0,
        width: float | None = None,
        height: float | None = None,
        background: RGBA = (0.0, 0.0, 0.0, 0.0),
        bar_color: RGBA | None = None,
    ) -> None: ...
    def scroll_to(self, x: float, y: float) -> None:
        """Scroll so content point `(x, y)` is at the top-left (clamped)."""

class Theme:
    """Named colors that widgets use when their own color arguments are left out, plus the
    colors chrome draws itself (window background, focus ring, scrollbars, modal dimming, dock
    drop preview). `Theme()` is the dark default with any keyword colors replaced."""
    background: RGBA
    surface: RGBA
    surface_alt: RGBA
    surface_active: RGBA
    border: RGBA
    divider: RGBA
    track: RGBA
    text: RGBA
    text_muted: RGBA
    accent: RGBA
    button: RGBA
    button_text: RGBA
    selection: RGBA
    scrollbar: RGBA
    scrim: RGBA
    drop_indicator: RGBA
    def __init__(self, **colors: RGBA) -> None: ...
    @staticmethod
    def dark() -> Theme: ...
    @staticmethod
    def light() -> Theme: ...
    def replace(self, **colors: RGBA) -> Theme:
        """A copy with the given colors changed."""

def set_theme(theme: Theme) -> None:
    """Make `theme` current for widgets described from now on (and chrome's next rebuild).
    Use `Window.set_theme` to restyle a window that's already showing."""

def get_theme() -> Theme: ...

class ListView:
    """A virtualized list of text rows: only the rows in view are drawn, so it handles millions.
    Click or arrow keys select (`on_select(index)`); double-click or Enter activates
    (`on_activate(index)`). Scrolls with the wheel/trackpad and a draggable scrollbar."""
    def __init__(
        self,
        items: Sequence[str],
        on_select: Callable[[int], None] | None = None,
        on_activate: Callable[[int], None] | None = None,
        row_height: float = 24.0,
        font_size: float = 14.0,
        flex_grow: float = 1.0,
        width: float | None = None,
        height: float | None = None,
        text_color: RGBA | None = None,
        background: RGBA | None = None,
        selection_color: RGBA | None = None,
    ) -> None: ...
    def set_items(self, items: Sequence[str]) -> None:
        """Replace the rows (clears the selection). Works before attaching."""
    def select(self, index: int | None) -> None:
        """Select a row (clamped) and scroll it into view, calling `on_select`."""
    @property
    def selected(self) -> int | None: ...
    def __len__(self) -> int: ...

class Popup:
    """An overlay shown on demand: a menu, dropdown list, tooltip or dialog. Not placed in the
    layout; open it with `show(anchor)` or `Window.show_popup`. A click outside a non-modal popup,
    or Escape, dismisses it (calling `on_dismiss`); a modal one dims the window and blocks clicks
    behind it. Tab stays inside the open popup; focus returns where it was when it closes. Drawn
    above everything, including `Viewport` video."""
    def __init__(
        self,
        content: Widget,
        modal: bool = False,
        on_dismiss: Callable[[], None] | None = None,
        padding: float = 6.0,
        background: RGBA | None = None,
        border: RGBA | None = None,
    ) -> None: ...
    def show(self, anchor: Widget, side: str = "below") -> None:
        """Open next to `anchor` (already shown in a window): "below", "above", "right" or "left",
        flipped when there's no room. Reopening moves it."""
    def close(self) -> None:
        """Close without calling `on_dismiss`."""
    @property
    def is_open(self) -> bool: ...

Widget = Union[Label, Button, Slider, TextInput, "ListView", "ScrollArea", "Box", "Splitter", "Panel", "Tabs", "DockArea", Viewport]

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
        bar_color: RGBA | None = None,
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
        title_color: RGBA | None = None,
        title_background: RGBA | None = None,
        background: RGBA | None = None,
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
        text_color: RGBA | None = None,
        active_color: RGBA | None = None,
        inactive_color: RGBA | None = None,
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
    def set_theme(self, theme: Theme) -> None:
        """Make `theme` current and rebuild this window's content so it shows at once."""
    def show_popup(self, popup: Popup, x: float | None = None, y: float | None = None) -> None:
        """Open `popup` with its top-left at `(x, y)`, or centered when both are omitted."""
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
