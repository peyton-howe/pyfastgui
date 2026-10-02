"""Tier 4 composites: image viewer, log, code editor, command palette, observables.

The drawing widgets (`Gauge`, `Timeline`, `NodeGraph`, plots) live in the extension module.
These classes are built from widgets that already exist.
"""

from __future__ import annotations

import threading

from ._fastgui import Box, Image, Label, ListView, Popup, Slider, TextArea, TextInput

_KEYWORDS = frozenset(
    """
    False None True and as assert async await break class continue def del elif else except
    finally for from global if import in is lambda nonlocal not or pass raise return try
    while with yield
    """.split()
)
_KEYWORD = (0.45, 0.72, 1.0, 1.0)
_COMMENT = (0.45, 0.50, 0.55, 1.0)
_STRING = (0.55, 0.80, 0.50, 1.0)
_NUMBER = (0.95, 0.70, 0.40, 1.0)


def highlight_python(text: str) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """Byte spans for keywords, comments, strings, and numbers."""
    spans: list[tuple[int, int, tuple[float, float, float, float]]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "#":
            j = text.find("\n", i)
            j = n if j < 0 else j
            spans.append((i, j, _COMMENT))
            i = j
            continue
        if c in ("'", '"'):
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    break
                j += 1
            if j < n and text[j] == c:
                j += 1
            spans.append((i, j, _STRING))
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] in ".eE"):
                j += 1
            spans.append((i, j, _NUMBER))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if text[i:j] in _KEYWORDS:
                spans.append((i, j, _KEYWORD))
            i = j
            continue
        i += 1
    return spans


def _clamp_span(a: float, b: float) -> tuple[float, float]:
    span = min(1.0, max(1e-4, abs(b - a)))
    mid = min(1.0 - span * 0.5, max(span * 0.5, (a + b) * 0.5))
    return mid - span * 0.5, mid + span * 0.5


def _zoom_window(window, tx: float, ty: float, factor: float):
    def zoom(a, b, t):
        t = min(1.0, max(0.0, t))
        factor_c = factor if factor > 1e-6 else 1.0
        anchor = a + t * (b - a)
        span = max(abs(b - a) / factor_c, 1e-12)
        return anchor - t * span, anchor + (1.0 - t) * span

    x0, x1 = _clamp_span(*zoom(window[0], window[1], tx))
    y0, y1 = _clamp_span(*zoom(window[2], window[3], ty))
    return (x0, x1, y0, y1)


def _pan_window(window, dx: float, dy: float):
    def shift(a, b, delta):
        amount = (b - a) * delta
        return a + amount, b + amount

    x0, x1 = _clamp_span(*shift(window[0], window[1], dx))
    y0, y1 = _clamp_span(*shift(window[2], window[3], dy))
    return (x0, x1, y0, y1)


class ImageViewer:
    """Zoom, pan, and read pixel values on a numpy image or 2-D tensor.

    The frame is letterboxed inside the upload buffer (the GPU layer then stretches that
    buffer). Wheel zooms, drag pans, double-click resets. Hover updates `readout` and does
    not re-rasterize. `colormap` is `gray`, `viridis`, or `magma` for 2-D arrays.
    """

    def __init__(
        self,
        array,
        colormap="gray",
        on_readout=None,
        pixel_width=480,
        pixel_height=270,
        width=None,
        height=None,
        flex_grow=1.0,
    ):
        self._array = array
        self.colormap = colormap
        self.on_readout = on_readout
        self._pw = int(pixel_width)
        self._ph = int(pixel_height)
        self._window = (0.0, 1.0, 0.0, 1.0)
        self._content = (0, 0, self._pw, self._ph)
        self.readout = ""
        self._lock = threading.Lock()
        self._image = Image(
            width=width,
            height=height,
            flex_grow=flex_grow,
            fit="stretch",
            on_pointer=self._pointer,
        )
        self._fastgui_widget = self._image
        self._render()

    def set_array(self, array):
        """Replace the source. Keeps the current zoom window."""
        with self._lock:
            self._array = array
        self._render()

    def reset_view(self):
        self._window = (0.0, 1.0, 0.0, 1.0)
        self._render()

    def _pointer(self, action, dx, dy, x, y, w, h):
        if action == 3:
            self._readout_at(x, y, w, h)
            return
        if action in (4, 5):
            return
        if action == 2:
            self._window = (0.0, 1.0, 0.0, 1.0)
        elif action == 0:
            factor = 1.15 if dy < 0.0 else 1.0 / 1.15
            fx = (x / w) if w else 0.5
            fy = (y / h) if h else 0.5
            self._window = _zoom_window(self._window, fx, fy, factor)
        elif action == 1:
            fx = (-dx / w) if w else 0.0
            fy = (dy / h) if h else 0.0
            self._window = _pan_window(self._window, fx, fy)
        else:
            return
        self._render()

    def _readout_at(self, x, y, w, h):
        import numpy as np

        array = np.asarray(self._array)
        if array.ndim < 2 or w <= 0 or h <= 0:
            return
        bx = int(x / w * self._pw)
        by = int(y / h * self._ph)
        cx, cy, cw, ch = self._content
        if cw <= 0 or ch <= 0 or not (cx <= bx < cx + cw and cy <= by < cy + ch):
            text = ""
        else:
            col0, col1, row0, row1 = self._window
            src_h, src_w = array.shape[:2]
            u = (bx - cx + 0.5) / cw
            v = (by - cy + 0.5) / ch
            sx = min(src_w - 1, max(0, int((col0 + u * (col1 - col0)) * src_w)))
            sy = min(src_h - 1, max(0, int((row0 + v * (row1 - row0)) * src_h)))
            sample = array[sy, sx]
            text = f"{sx},{sy}: {sample}"
        self.readout = text
        if self.on_readout is not None and text:
            self.on_readout(text)

    def _render(self):
        import numpy as np

        with self._lock:
            array = np.asarray(self._array)
        if array.ndim == 2:
            src = _colorize(array, self.colormap)
        elif array.ndim == 3 and array.shape[2] in (3, 4):
            src = np.ascontiguousarray(array)
            if src.dtype != np.uint8:
                src = np.clip(src, 0, 255).astype(np.uint8)
            if src.shape[2] == 3:
                alpha = np.full(src.shape[:2] + (1,), 255, np.uint8)
                src = np.concatenate([src, alpha], axis=2)
        else:
            raise ValueError("ImageViewer expects a 2-D array or an HxWx3/4 image")
        src_h, src_w = src.shape[:2]
        col0, col1, row0, row1 = self._window
        win_w = max((col1 - col0) * src_w, 1e-6)
        win_h = max((row1 - row0) * src_h, 1e-6)
        aspect = win_w / win_h
        box_aspect = self._pw / max(self._ph, 1)
        if aspect > box_aspect:
            content_w = self._pw
            content_h = max(int(round(self._pw / aspect)), 1)
        else:
            content_h = self._ph
            content_w = max(int(round(self._ph * aspect)), 1)
        content_h = min(content_h, self._ph)
        content_w = min(content_w, self._pw)
        x0 = (self._pw - content_w) // 2
        y0 = (self._ph - content_h) // 2
        self._content = (x0, y0, content_w, content_h)
        buf = np.zeros((self._ph, self._pw, 4), np.uint8)
        buf[:, :, 3] = 255
        xs = np.linspace(col0, col1, content_w, endpoint=False)
        ys = np.linspace(row0, row1, content_h, endpoint=False)
        cols = np.clip((xs * src_w).astype(np.int32), 0, src_w - 1)
        rows = np.clip((ys * src_h).astype(np.int32), 0, src_h - 1)
        buf[y0 : y0 + content_h, x0 : x0 + content_w] = src[rows][:, cols]
        self._image.set_image(np.ascontiguousarray(buf))


def _colorize(values, name: str):
    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        norm = np.zeros(data.shape, np.float64)
    else:
        norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    t = norm
    name = (name or "gray").lower()
    if name in ("gray", "grey"):
        g = (t * 255.0).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    elif name == "magma":
        r = np.clip(0.001 + 2.2 * t - 1.3 * t * t, 0.0, 1.0)
        g = np.clip(0.0 + 0.2 * t + 1.4 * t * t - 0.7 * t * t * t, 0.0, 1.0)
        b = np.clip(0.016 + 1.5 * t - 1.8 * t * t + 0.9 * t * t * t, 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    else:
        # A short viridis-like ramp (purple → teal → yellow).
        r = np.clip(0.27 + 0.1 * t + 0.7 * t * t, 0.0, 1.0)
        g = np.clip(0.0 + 1.1 * t - 0.15 * t * t, 0.0, 1.0)
        b = np.clip(0.33 + 0.6 * t - 0.9 * t * t, 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    alpha = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


class LogView:
    """Append-only console. `append` is safe from any thread; callers do not take a lock."""

    def __init__(self, max_lines=1000, flex_grow=1.0, height=None, width=None):
        self.max_lines = int(max_lines)
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._list = ListView([], flex_grow=flex_grow, height=height, width=width)
        self._fastgui_widget = self._list

    def append(self, line: str) -> None:
        text = str(line).rstrip("\n")
        with self._lock:
            self._lines.extend(text.split("\n"))
            if len(self._lines) > self.max_lines:
                self._lines = self._lines[-self.max_lines :]
            items = list(self._lines)
        self._list.set_items(items)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
        self._list.set_items([])

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)


class CodeEditor:
    """`TextArea` that recolors keywords, comments, strings, and numbers on each edit."""

    def __init__(self, text="", placeholder="", width=None, height=None, flex_grow=1.0, on_change=None):
        self._user_change = on_change

        def changed(value: str) -> None:
            self._area.set_highlights(highlight_python(value))
            if self._user_change is not None:
                self._user_change(value)

        self._area = TextArea(
            text,
            placeholder=placeholder,
            on_change=changed,
            width=width,
            height=height,
            flex_grow=flex_grow,
        )
        self._fastgui_widget = self._area
        self._area.set_highlights(highlight_python(text))

    @property
    def text(self) -> str:
        return self._area.text

    def set_text(self, text: str) -> None:
        self._area.set_text(text)
        self._area.set_highlights(highlight_python(text))


class CommandPalette:
    """Ctrl+K palette: a modal popup with a filter field and a command list.

    Call `bind(anchor)` with a widget that is in the window, and put `accelerators()` on
    that window's root `Box`. `show()` opens it at the top of the window.
    """

    def __init__(self, commands):
        self._commands = [(str(name), fn) for name, fn in commands]
        self._shown = list(self._commands)
        self._anchor = None
        self._filter = TextInput(placeholder="Type a command…", on_change=self._apply)
        self._list = ListView([name for name, _ in self._shown], on_activate=self._run, height=240.0)
        self._popup = Popup(
            Box(
                direction="column",
                gap=6.0,
                padding=8.0,
                width=360.0,
                children=[self._filter, self._list],
            ),
            modal=True,
        )

    def bind(self, anchor) -> None:
        self._anchor = anchor

    def accelerators(self):
        return [("Ctrl+K", self.show)]

    def show(self) -> None:
        if self._anchor is None:
            return
        self._filter.set_text("")
        self._apply("")
        self._popup.show_at(self._anchor, 80.0, 48.0)

    def _apply(self, query: str) -> None:
        q = query.strip().lower()
        self._shown = [item for item in self._commands if q in item[0].lower()]
        self._list.set_items([name for name, _ in self._shown])

    def _run(self, index: int) -> None:
        if not 0 <= index < len(self._shown):
            return
        _name, fn = self._shown[index]
        self._popup.close()
        fn()


class Observable:
    """A value plus subscribers. Widget setters used here do not echo back into `set`."""

    def __init__(self, value):
        self._value = value
        self._subs: list = []
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._value

    def set(self, value) -> None:
        with self._lock:
            self._value = value
            subs = list(self._subs)
        for fn in subs:
            fn(value)

    def subscribe(self, fn) -> None:
        with self._lock:
            self._subs.append(fn)
        fn(self.get())

    def bind_label(self, fmt=None):
        label = Label(self._format(self.get(), fmt))

        def update(value, label=label, fmt=fmt):
            label.set_text(self._format(value, fmt))

        self.subscribe(update)
        return label

    def bind_text(self, widget):
        """Push this value into an existing `TextInput` or anything with `set_text`."""

        def push(value, widget=widget):
            text = value if isinstance(value, str) else str(value)
            current = getattr(widget, "text", None)
            if current != text:
                widget.set_text(text)

        self.subscribe(push)
        return widget

    def text_input(self, **kwargs):
        user = kwargs.pop("on_change", None)
        field = TextInput(self._format(self.get(), None), on_change=lambda text: self._from_text(text, user), **kwargs)

        def push(value, field=field):
            text = value if isinstance(value, str) else str(value)
            if field.text != text:
                field.set_text(text)

        self.subscribe(push)
        return field

    def slider(self, min=0.0, max=1.0, **kwargs):
        user = kwargs.pop("on_change", None)

        def on_change(value, user=user):
            if user is not None:
                user(value)
            if float(self.get()) != float(value):
                self.set(float(value))

        slider = Slider(value=float(self.get()), min=min, max=max, on_change=on_change, **kwargs)

        def push(value, slider=slider):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return
            slider.set_value(numeric)

        self.subscribe(push)
        return slider

    def _from_text(self, text, user):
        if user is not None:
            user(text)
        if str(self.get()) != text:
            self.set(text)

    @staticmethod
    def _format(value, fmt):
        if fmt is None:
            return value if isinstance(value, str) else str(value)
        return fmt(value)
