from typing import Any, Callable, Literal, Sequence, Union

RGBA = tuple[float, float, float, float]
# A font size in points, or a theme size by name.
FontSize = Union[float, Literal["small", "body", "large"]]
# A gap/padding in layout units, or a theme spacing by name.
Spacing = Union[float, Literal["small", "medium", "large"]]

# A CUDA stream: an int CUstream handle, torch.cuda.Stream, a cupy stream, or any object with
# __cuda_stream__().
CudaStream = Any

class CudaFrame:
    """One frame of a CudaSurface. Use as a context manager; inside the block it exposes
    __cuda_array_interface__ ((height, width, 4) uint8, packed rows), so
    torch.as_tensor(frame, device="cuda") or cupy.asarray(frame) wrap it without a copy. An
    exception in the block drops the frame. Unverified on NVIDIA hardware."""

    def __enter__(self) -> CudaFrame: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...

class CudaSurface:
    """Zero-copy CUDA target behind a Viewport, from Viewport.create_cuda_surface. Unverified on
    NVIDIA hardware."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def closed(self) -> bool:
        """True once the window dropped the surface (closed, viewport removed, another
        create_cuda_surface, or a submit_frame). frame() raises from then on."""
        ...
    def frame(self, stream: CudaStream | None = None) -> CudaFrame:
        """A frame written on `stream` (default: the legacy default stream). Entering waits,
        GPU-side on `stream`, for a free slot; leaving publishes what `stream` has written by
        then."""
        ...

class Viewport:
    def __init__(self) -> None: ...
    def submit_frame(self, data: Any) -> None:
        """Submit a (H, W, 3|4) uint8 frame. Copies into an owned buffer today; a packed-RGBA
        high-FPS path will want fewer copies later."""
        ...
    def submit_cuda(self, array: Any, stream: CudaStream | None = None) -> None:
        """Submit a CUDA array (anything with __cuda_array_interface__: torch, cupy, numba,
        jax) shaped (H, W, 4) uint8 with contiguous pixels. Copied after the work queued on
        `stream` (default: the array's own), and `stream` waits for the copy, so the array can
        be reused right away. One GPU-to-GPU copy on Windows/Vulkan/NVIDIA; elsewhere (macOS,
        Linux for now, a non-NVIDIA window GPU, before run()) a copy through host memory.
        Host arrays go to submit_frame. Unverified on NVIDIA hardware."""
        ...
    @property
    def cuda_status(self) -> str:
        """How the last submit_cuda reached the screen: "interop", "host copy: <reason>", or
        "unused"."""
        ...
    def create_cuda_surface(self, width: int, height: int) -> CudaSurface:
        """Zero-copy CUDA surface (Windows/Vulkan/NVIDIA). Call after set_viewport(), once
        window.run() has started (from a callback or another thread); raises RuntimeError if
        run() hasn't started within ~2s, or where interop isn't available (submit_cuda works
        everywhere)."""
        ...

class Label:
    def __init__(
        self,
        text: str,
        font_size: FontSize | None = None,
        color: RGBA | None = None,
        tooltip: str | None = None,
        context_menu: Any = None,
    ) -> None: ...
    def set_text(self, text: str) -> None: ...

class Button:
    def __init__(
        self,
        text: str,
        on_click: Callable[[], None] | None = None,
        font_size: FontSize | None = None,
        text_color: RGBA | None = None,
        background: RGBA | None = None,
        flat: bool = False,
        tooltip: str | None = None,
        context_menu: Any = None,
        on_hover: Callable[[], None] | None = None,
    ) -> None: ...
    def set_text(self, text: str) -> None: ...
    def set_background(self, color: RGBA | None = None) -> None:
        """Update fill; `None` clears to transparent (menu-title open highlight)."""
        ...

class Slider:
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 1.0,
        on_change: Callable[[float], None] | None = None,
        track_color: RGBA | None = None,
        thumb_color: RGBA | None = None,
        tooltip: str | None = None,
    ) -> None: ...
    def set_value(self, value: float) -> None: ...
    @property
    def value(self) -> float: ...

class TextInput:
    """A single-line editable text field. Click or Tab to focus; supports selection, word
    motion, undo/redo, the system clipboard and IME input."""
    def __init__(
        self,
        text: str = "",
        placeholder: str = "",
        on_change: Callable[[str], None] | None = None,
        on_submit: Callable[[str], None] | None = None,
        font_size: FontSize | None = None,
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

class TextArea:
    """A multi-line editable text field. Enter inserts a hard newline; Cmd/Ctrl+Enter fires
    `on_submit`. Soft wrap is not implemented yet — long lines are clipped."""
    def __init__(
        self,
        text: str = "",
        placeholder: str = "",
        on_change: Callable[[str], None] | None = None,
        on_submit: Callable[[str], None] | None = None,
        font_size: FontSize | None = None,
        width: float | None = None,
        height: float | None = None,
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
    def set_highlights(self, spans: Sequence[tuple[int, int, RGBA]]) -> None:
        """Color byte spans `(start, end, rgba)`. Empty clears them. The caret still uses the
        full line's glyph stops."""

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
    """Named colors, font sizes and spacings. Widgets use them when their own arguments are left
    out (colors, font sizes) or given by name (`font_size="large"`, `gap="medium"`), and chrome
    draws its own colors and all text with them. `Theme()` is the dark default with any keyword
    tokens replaced."""
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
    font_family: str | None
    font_size_small: float
    font_size: float
    font_size_large: float
    spacing_small: float
    spacing: float
    spacing_large: float
    def __init__(self, **tokens: Any) -> None: ...
    @staticmethod
    def dark() -> Theme: ...
    @staticmethod
    def light() -> Theme: ...
    def replace(self, **tokens: Any) -> Theme:
        """A copy with the given tokens changed (colors, font sizes/family, spacings)."""

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
        font_size: FontSize | None = None,
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
        padding: Spacing = 6.0,
        background: RGBA | None = None,
        border: RGBA | None = None,
        click_through: bool = False,
        closes_on_anchor_click: bool = True,
    ) -> None: ...
    def show(self, anchor: Widget, side: str = "below") -> None:
        """Open next to `anchor` (already shown in a window): "below", "above", "right" or "left",
        flipped when there's no room. Reopening moves it."""
    def show_at(self, near: Widget, x: float, y: float) -> None:
        """Open at window point `(x, y)` in the same window as attached widget `near`."""
    def close(self) -> None:
        """Close without calling `on_dismiss`."""
    @property
    def is_open(self) -> bool: ...

class Checkbox:
    """Labeled on/off checkbox. Click or Space/Enter toggles; `set_checked` does not fire `on_change`."""
    def __init__(
        self,
        label: str,
        checked: bool = False,
        on_change: Callable[[bool], None] | None = None,
        font_size: FontSize | None = None,
        text_color: RGBA | None = None,
        box_color: RGBA | None = None,
        check_color: RGBA | None = None,
        tooltip: str | None = None,
    ) -> None: ...
    @property
    def checked(self) -> bool: ...
    def set_checked(self, checked: bool) -> None: ...

class Radio:
    """One option in a radio group. Pass the same `group` id to peers so only one can be selected.
    Automatic groups use ids with the high bit set; user-supplied `group` must be `< 2**63`."""
    def __init__(
        self,
        label: str,
        group: int | None = None,
        selected: bool = False,
        on_select: Callable[[], None] | None = None,
        font_size: FontSize | None = None,
        text_color: RGBA | None = None,
        box_color: RGBA | None = None,
        dot_color: RGBA | None = None,
    ) -> None: ...
    @property
    def group(self) -> int: ...
    @property
    def selected(self) -> bool: ...
    def set_selected(self, selected: bool) -> None:
        """Select or clear without firing `on_select`. Selecting clears peers in the same group."""

class Toggle:
    def __init__(
        self,
        checked: bool = False,
        on_change: Callable[[bool], None] | None = None,
        track_off: RGBA | None = None,
        track_on: RGBA | None = None,
        thumb_color: RGBA | None = None,
        tooltip: str | None = None,
    ) -> None: ...
    @property
    def checked(self) -> bool: ...
    def set_checked(self, checked: bool) -> None: ...

class MenuItem:
    """One row of a `Menu`. `shortcut` ("Cmd+S", "Ctrl+Shift+N", "F5", "Del", ...; Cmd and Ctrl
    both mean the platform's primary modifier) is shown and registered on the `MenuBar`; an
    unsupported one raises ValueError. `checked` (bool) makes it a check item, `radio_group` +
    `checked` a radio item; `submenu` opens a nested `Menu` (on hover or click)."""
    def __init__(
        self,
        label: str,
        shortcut: str | None = None,
        on_click: Callable[[], None] | None = None,
        enabled: bool = True,
        checked: bool | None = None,
        radio_group: Any = None,
        icon: str | None = None,
        submenu: "Menu | None" = None,
    ) -> None: ...
    label: str
    shortcut: str | None
    on_click: Callable[[], None] | None
    enabled: bool
    checked: bool | None
    radio_group: Any
    icon: str | None
    submenu: "Menu | None"

class MenuSeparator:
    def __init__(self) -> None: ...

class Menu:
    """Popup menu of `MenuItem` / `MenuSeparator` rows."""
    def __init__(self, items: Sequence[MenuItem | MenuSeparator]) -> None: ...
    def as_popup(
        self,
        on_dismiss: Callable[[], None] | None = None,
        click_through: bool = False,
        closes_on_anchor_click: bool = True,
    ) -> Popup: ...
    def show(
        self,
        anchor: Widget,
        side: str = "below",
        on_dismiss: Callable[[], None] | None = None,
        click_through: bool = False,
        closes_on_anchor_click: bool = True,
    ) -> None: ...
    def show_at(self, near: Widget, x: float, y: float, on_dismiss: Callable[[], None] | None = None) -> None: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...

class MenuBar:
    """Traditional menu strip: flat titles, hover/open highlight, thin bottom edge.

    `menus` is a sequence of `(title, Menu)`.
    """
    def __init__(self, menus: Sequence[tuple[str, Menu]]) -> None: ...

class Toolbar:
    """Horizontal strip for Buttons/Labels/spacers."""
    def __init__(self, children: Sequence["Widget"]) -> None: ...

class StatusBar:
    """Bottom status strip; `set_text` updates the muted label."""
    def __init__(self, text: str = "") -> None: ...
    def set_text(self, text: str) -> None: ...

class Dialog:
    """Modal/modeless wrapper around `Popup`. `buttons` is `(label, on_click)` pairs."""
    def __init__(
        self,
        title: str,
        content: "Widget",
        buttons: Sequence[tuple[str, Callable[[], None] | None]] | None = None,
        modal: bool = True,
        on_dismiss: Callable[[], None] | None = None,
    ) -> None: ...
    def as_popup(self) -> Popup: ...
    def show(self, window: "Window") -> None: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...

def open_file_dialog(
    title: str | None = None,
    filter: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> str | None:
    """Native open-file dialog (rfd). Returns a path or `None` if cancelled."""
    ...

def save_file_dialog(
    title: str | None = None,
    filter: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> str | None:
    """Native save-file dialog (rfd). Returns a path or `None` if cancelled."""
    ...

def pick_color(
    window: "Window",
    initial: RGBA | None = None,
    on_pick: Callable[[RGBA | None], None] | None = None,
) -> Dialog:
    """In-app RGB picker (rfd has no color dialog). Calls `on_pick` on OK/Cancel."""
    ...

class GroupBox:
    """Titled bordered frame around `content`."""
    def __init__(self, title: str, content: "Widget", padding: Spacing = "medium") -> None: ...

class CollapsibleSection:
    """Header toggles body visibility."""
    def __init__(self, title: str, content: "Widget", expanded: bool = True) -> None: ...
    def set_expanded(self, expanded: bool) -> None: ...
    @property
    def expanded(self) -> bool: ...

class StackedWidget:
    """Only one child visible; `set_index` toggles the rest off."""
    def __init__(self, children: Sequence["Widget"], index: int = 0) -> None: ...
    @property
    def index(self) -> int: ...
    def set_index(self, index: int) -> None: ...

class SpinBox:
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 100.0,
        step: float = 1.0,
        decimals: int = 0,
        on_change: Callable[[float], None] | None = None,
        font_size: FontSize | None = None,
        width: float | None = None,
        text_color: RGBA | None = None,
        background: RGBA | None = None,
        button_color: RGBA | None = None,
    ) -> None: ...
    @property
    def value(self) -> float: ...
    def set_value(self, value: float) -> None: ...

class NumericScrub:
    """Drag horizontally to change the value (`speed` units per logical pixel)."""
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 100.0,
        speed: float = 0.25,
        decimals: int = 1,
        on_change: Callable[[float], None] | None = None,
        font_size: FontSize | None = None,
        width: float | None = None,
        text_color: RGBA | None = None,
        background: RGBA | None = None,
    ) -> None: ...
    @property
    def value(self) -> float: ...
    def set_value(self, value: float) -> None: ...

class ProgressBar:
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 1.0,
        track_color: RGBA | None = None,
        fill_color: RGBA | None = None,
        height: float = 8.0,
        flex_grow: float = 1.0,
    ) -> None: ...
    @property
    def value(self) -> float: ...
    def set_value(self, value: float) -> None: ...

class ComboBox:
    """Closed dropdown field. Click / Space / ArrowDown opens a popup list; choosing a row
    (or activating) sets the selection, fires `on_change(index)`, and closes the popup."""
    def __init__(
        self,
        items: Sequence[str] | None = None,
        selected: int | None = None,
        placeholder: str = "",
        on_change: Callable[[int], None] | None = None,
        font_size: FontSize | None = None,
        width: float | None = None,
        flex_grow: float = 0.0,
        text_color: RGBA | None = None,
        placeholder_color: RGBA | None = None,
        background: RGBA | None = None,
        border: RGBA | None = None,
    ) -> None: ...
    def set_items(self, items: Sequence[str]) -> None:
        """Replace the rows (clears the selection). Works before attaching."""
    def select(self, index: int | None) -> None:
        """Select a row (clamped) or clear with `None`."""
    @property
    def selected(self) -> int | None: ...

class Image:
    """Static CPU image composited like a `Viewport`. Feed pixels with `set_image`."""
    def __init__(
        self,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        fit: str = "stretch",
        on_pointer: Callable[[int, float, float, float, float, float, float], None] | None = None,
    ) -> None: ...
    def set_image(self, data: Any) -> None:
        """Submit a (H, W, 3|4) uint8 array."""
        ...

class Grid:
    """CSS Grid with `columns` equal-width tracks; children auto-flow into rows."""
    def __init__(
        self,
        children: Sequence["Widget"],
        columns: int = 2,
        gap: Spacing = 0.0,
        padding: Spacing = 0.0,
        flex_grow: float = 0.0,
        width: float | None = None,
        height: float | None = None,
        background: RGBA = (0.0, 0.0, 0.0, 0.0),
    ) -> None: ...

Widget = Union[
    Label,
    Button,
    Slider,
    TextInput,
    TextArea,
    "ListView",
    "ScrollArea",
    "Popup",
    Checkbox,
    Radio,
    Toggle,
    SpinBox,
    NumericScrub,
    ProgressBar,
    ComboBox,
    Image,
    Grid,
    "Box",
    "Splitter",
    "Panel",
    "Tabs",
    "DockArea",
    "Toolbar",
    "StatusBar",
    "Dialog",
    "GroupBox",
    "CollapsibleSection",
    "StackedWidget",
    "MenuBar",
    Viewport,
]

class Box:
    def __init__(
        self,
        children: Sequence[Widget],
        direction: str = "column",
        gap: Spacing = 0.0,
        padding: Spacing = 0.0,
        flex_grow: float = 0.0,
        width: float | None = None,
        height: float | None = None,
        background: RGBA = (0.0, 0.0, 0.0, 0.0),
        wrap: bool = False,
        visible: bool = True,
        context_menu: Any = None,
        accelerators: Sequence[tuple[str, Callable[[], None]]] | None = None,
    ) -> None: ...
    def set_display(self, visible: bool) -> None:
        """Show or hide this box (`Display::None` when hidden)."""
        ...
    def set_background(self, color: RGBA | None = None) -> None:
        """Change the fill color (`None`: transparent). Works before showing; survives rebuilds."""


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
        title_font_size: FontSize | None = None,
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
        font_size: FontSize | None = None,
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
    def show_popup(self, popup: Popup | Dialog | Any, x: float | None = None, y: float | None = None) -> None:
        """Open a `Popup`/`Dialog` (or `as_popup()` object) at `(x, y)`, or centered when omitted."""
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

class PlotLine:
    """Polyline plot rasterized to an RGBA image layer. Wheel zoom, drag pan, double-click reset
    when `interactive` (the default). `set_data` keeps the current view."""
    def __init__(
        self,
        x: Any = None,
        y: Any = None,
        *,
        color: RGBA | None = None,
        background: RGBA | None = None,
        thickness: float = 2.0,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, x: Any, y: Any) -> None: ...
    def set_series(self, series: Sequence[Any]) -> None: ...
    def set_range(self, x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view_range(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

class PlotScatter:
    """Scatter plot. Same interaction and `set_series` shape as `PlotLine`."""
    def __init__(
        self,
        x: Any = None,
        y: Any = None,
        *,
        color: RGBA | None = None,
        background: RGBA | None = None,
        point_radius: float = 3.0,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, x: Any, y: Any) -> None: ...
    def set_series(self, series: Sequence[Any]) -> None: ...
    def set_range(self, x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view_range(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

class PlotHeatmap:
    """2-D heatmap (`viridis`, `magma`, or `gray`). Pan/zoom moves a window over the grid."""
    def __init__(
        self,
        values: Any = None,
        *,
        colormap: str = "viridis",
        background: RGBA | None = None,
        v_range: tuple[float, float] | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, values: Any) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def window(self) -> tuple[float, float, float, float]: ...

class PlotHistogram:
    """Histogram of a 1-D sample. `bin_range` drops values outside that domain. Wheel/drag/double-click."""
    def __init__(
        self,
        values: Any = None,
        *,
        bins: int = 20,
        bin_range: tuple[float, float] | None = None,
        color: RGBA | None = None,
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, values: Any) -> None: ...
    def set_bins(self, bins: int) -> None: ...
    def counts(self) -> list[float]: ...
    def edges(self) -> list[float]: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view_range(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

class PlotBar:
    """Vertical bars from a baseline of 0. `heights[i]` is category `i`."""
    def __init__(
        self,
        heights: Any = None,
        *,
        color: RGBA | None = None,
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, heights: Any) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view_range(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

class PlotContour:
    """Isolines of a 2-D grid. `levels` is a count or a sequence of values. `filled` paints bands."""
    def __init__(
        self,
        values: Any = None,
        *,
        levels: int | Sequence[float] | None = None,
        filled: bool = False,
        colormap: str = "viridis",
        background: RGBA | None = None,
        v_range: tuple[float, float] | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, values: Any) -> None: ...
    def set_levels(self, levels: int | Sequence[float]) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def window(self) -> tuple[float, float, float, float]: ...

class PlotSurface:
    """3-D surface. `z` is a height grid. Optional `x` and `y` are axes or grids of the same shape. Drag orbits."""
    def __init__(
        self,
        z: Any = None,
        *,
        x: Any = None,
        y: Any = None,
        colormap: str = "viridis",
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, z: Any, x: Any = None, y: Any = None) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view(self) -> tuple[float, float, float]: ...

class PlotScatter3D:
    """3-D scatter. Pass `x, y, z` or one `(N, 3)` array. Drag orbits, wheel zooms. Colored by z."""
    def __init__(
        self,
        x: Any = None,
        y: Any = None,
        z: Any = None,
        *,
        point_radius: float = 3.5,
        colormap: str = "viridis",
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, x: Any, y: Any = None, z: Any = None) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view(self) -> tuple[float, float, float]: ...

class PlotMesh:
    """Triangle mesh. `vertices` is `(N, 3)`, `faces` is `(M, 3)` indexes. Drag orbits."""
    def __init__(
        self,
        vertices: Any = None,
        faces: Sequence[tuple[int, int, int]] | None = None,
        *,
        colormap: str = "viridis",
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        interactive: bool = True,
    ) -> None: ...
    def set_data(self, vertices: Any, faces: Sequence[tuple[int, int, int]]) -> None: ...
    def zoom(self, factor: float, fx: float, fy: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def reset_view(self) -> None: ...
    def view(self) -> tuple[float, float, float]: ...

class Gauge:
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 1.0,
        *,
        color: RGBA | None = None,
        background: RGBA | None = None,
        pixel_width: int = 320,
        pixel_height: int = 180,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
    ) -> None: ...
    @property
    def value(self) -> float: ...
    def set_value(self, value: float) -> None: ...

class Timeline:
    def __init__(
        self,
        tracks: Sequence[Any] | None = None,
        duration: float = 10.0,
        time: float = 0.0,
        *,
        color: RGBA | None = None,
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 160,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
    ) -> None: ...
    @property
    def time(self) -> float: ...
    def set_time(self, time: float) -> None: ...
    def set_duration(self, duration: float) -> None: ...
    def set_tracks(self, tracks: Sequence[Any]) -> None: ...

class NodeGraph:
    """Draggable nodes. Each node is `(x, y, title)` or `(x, y, w, h, title)` in frame pixels."""
    def __init__(
        self,
        nodes: Sequence[Any] | None = None,
        edges: Sequence[tuple[int, int]] | None = None,
        *,
        color: RGBA | None = None,
        background: RGBA | None = None,
        pixel_width: int = 640,
        pixel_height: int = 360,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
    ) -> None: ...
    def set_nodes(self, nodes: Sequence[Any]) -> None: ...
    def set_edges(self, edges: Sequence[tuple[int, int]]) -> None: ...
    def nodes(self) -> list[tuple[float, float, float, float, str, bool]]: ...

class ImageViewer:
    def __init__(
        self,
        array: Any,
        colormap: str = "gray",
        on_readout: Callable[[str], None] | None = None,
        pixel_width: int = 480,
        pixel_height: int = 270,
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
    ) -> None: ...
    readout: str
    def set_array(self, array: Any) -> None: ...
    def reset_view(self) -> None: ...

class LogView:
    def __init__(self, max_lines: int = 1000, flex_grow: float = 1.0, height: float | None = None, width: float | None = None) -> None: ...
    def append(self, line: str) -> None: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...

class CodeEditor:
    def __init__(
        self,
        text: str = "",
        placeholder: str = "",
        width: float | None = None,
        height: float | None = None,
        flex_grow: float = 1.0,
        on_change: Callable[[str], None] | None = None,
    ) -> None: ...
    @property
    def text(self) -> str: ...
    def set_text(self, text: str) -> None: ...

class CommandPalette:
    def __init__(self, commands: Sequence[tuple[str, Callable[[], None]]]) -> None: ...
    def bind(self, anchor: Any) -> None: ...
    def accelerators(self) -> list[tuple[str, Callable[[], None]]]: ...
    def show(self) -> None: ...

class Observable:
    def __init__(self, value: Any) -> None: ...
    def get(self) -> Any: ...
    def set(self, value: Any) -> None: ...
    def subscribe(self, fn: Callable[[Any], None]) -> None: ...
    def bind_label(self, fmt: Callable[[Any], str] | None = None) -> Label: ...
    def bind_text(self, widget: Any) -> Any: ...
    def text_input(self, **kwargs: Any) -> TextInput: ...
    def slider(self, min: float = 0.0, max: float = 1.0, **kwargs: Any) -> Slider: ...
