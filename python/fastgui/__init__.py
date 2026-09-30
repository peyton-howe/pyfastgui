from ._fastgui import (
    Box,
    Button,
    Checkbox,
    ComboBox,
    CudaSurface,
    Grid,
    Image,
    Label,
    ListView,
    NumericScrub,
    Panel,
    Popup,
    ProgressBar,
    Radio,
    ScrollArea,
    Slider,
    SpinBox,
    Splitter,
    Tabs,
    TextArea,
    TextInput,
    Theme,
    Toggle,
    Viewport,
    Window,
    get_theme,
    open_file_dialog,
    save_file_dialog,
    set_theme,
)

_REGIONS = ("center", "left", "right", "top", "bottom")

# Matches `fastgui-app`'s `ROOT_REGION_ID` — sent as `target_id` when a panel is
# dropped near the *window's* outer edge (not a specific panel's), meaning "wrap the whole
# DockArea", not "split just whatever panel happens to be under the cursor". Never a real
# `Panel`/`Tabs` id (`next_region_id()`, Rust-side, starts at 1).
_ROOT_REGION_ID = 0


class MenuItem:
    """One actionable row in a `Menu`.

    Optional `checked` draws a leading checkmark (True) or an aligned blank (False).
    Optional `radio_group` (string/id) draws ●/○ and groups exclusivity for the app to manage
    on `on_click`. Optional `icon` is a short leading glyph. Optional `submenu` opens a nested
    `Menu` to the right instead of firing `on_click` / closing the parent.
    """

    def __init__(
        self,
        label,
        shortcut=None,
        on_click=None,
        enabled=True,
        checked=None,
        radio_group=None,
        icon=None,
        submenu=None,
    ):
        self.label = label
        self.shortcut = shortcut
        self.on_click = on_click
        self.enabled = bool(enabled)
        self.checked = checked
        self.radio_group = radio_group
        self.icon = icon
        self.submenu = submenu
        if submenu is not None and not isinstance(submenu, Menu):
            raise TypeError(f"submenu must be a Menu, got {type(submenu)!r}")
        if shortcut and submenu is not None:
            raise ValueError("MenuItem cannot have both shortcut and submenu")


class MenuSeparator:
    """A thin muted bar between `MenuItem`s."""


class Menu:
    """A popup column of `MenuItem` / `MenuSeparator` rows. Open with `show(anchor)` or
    `show_at(near, x, y)`; also usable as `context_menu=` (via `as_popup()`)."""

    def __init__(self, items):
        self.items = list(items)
        self._popup = None
        self._open_submenu = None
        # The menu this one is open as a submenu of (None for a top-level menu).
        self._parent = None
        # Cleared after running once. Fired both on outside/Escape dismiss and on `close()`
        # (menu-item clicks use `Popup.close`, which does not call the Popup's on_dismiss).
        self._on_dismiss = None

    def _item_label(self, item):
        parts = []
        if item.radio_group is not None:
            parts.append("●" if item.checked else "○")
        elif item.checked is not None:
            parts.append("✓" if item.checked else " ")
        if item.icon:
            parts.append(str(item.icon))
        parts.append(item.label)
        text = "  ".join(parts)
        if item.submenu is not None:
            return f"{text}    ▶"
        if item.shortcut:
            return f"{text}    {item.shortcut}"
        return text

    def _close_submenu(self):
        if self._open_submenu is not None:
            self._open_submenu.close()
            self._open_submenu = None

    def _toggle_submenu(self, submenu, anchor_button):
        if self._open_submenu is submenu and submenu.is_open:
            self._close_submenu()
            return
        self._close_submenu()
        self._open_submenu = submenu
        submenu.show(anchor_button, side="right", on_dismiss=lambda: setattr(self, "_open_submenu", None))
        submenu._parent = self

    def _root(self):
        menu = self
        while menu._parent is not None:
            menu = menu._parent
        return menu

    def _build_row(self, item):
        theme = get_theme()
        if isinstance(item, MenuSeparator):
            return Box(
                direction="column",
                height=9.0,
                padding=4.0,
                children=[
                    Box(
                        direction="row",
                        height=1.0,
                        flex_grow=1.0,
                        background=theme.border,
                        children=[],
                    )
                ],
            )
        if not isinstance(item, MenuItem):
            raise TypeError(f"Menu items must be MenuItem or MenuSeparator, got {type(item)!r}")

        holder = {"btn": None}

        def activate(it=item, h=holder):
            if not it.enabled:
                return
            if it.submenu is not None:
                self._toggle_submenu(it.submenu, h["btn"])
                return
            # A pick anywhere in a submenu chain closes the whole chain, from the top menu (whose
            # on_dismiss also clears e.g. the MenuBar title highlight).
            self._root().close()
            if it.on_click is not None:
                it.on_click()

        label_color = theme.text if item.enabled else theme.text_muted
        # Flat rows: transparent until hover (`surface_active`), like a native menu.
        btn = Button(
            self._item_label(item),
            on_click=activate if item.enabled else None,
            font_size="small",
            text_color=label_color,
            background=None,
            flat=True,
        )
        holder["btn"] = btn
        return btn

    def _take_dismiss(self):
        callback = self._on_dismiss
        self._on_dismiss = None
        return callback

    def as_popup(self, on_dismiss=None, click_through=False):
        """Fresh `Popup` for `context_menu=` / `Window.show_popup` (rebuilds each call).
        `click_through`: a click outside that closes it also reaches the widget there (a
        `MenuBar` uses it so clicking another title switches menus in one click)."""
        self._close_submenu()
        self._parent = None
        rows = [self._build_row(item) for item in self.items]

        def wrapped_dismiss(cb=on_dismiss):
            # Outside click / Escape: Popup fires this; drop our copy so `close()` won't re-run it.
            self._close_submenu()
            self._on_dismiss = None
            if cb is not None:
                cb()

        self._on_dismiss = on_dismiss
        self._popup = Popup(
            Box(direction="column", gap=0.0, children=rows),
            padding=4.0,
            on_dismiss=wrapped_dismiss if on_dismiss is not None else None,
            click_through=click_through,
        )
        return self._popup

    def show(self, anchor, side="below", on_dismiss=None, click_through=False):
        self.as_popup(on_dismiss=on_dismiss, click_through=click_through).show(anchor, side=side)

    def show_at(self, near, x, y, on_dismiss=None):
        """Open at window point `(x, y)` using `near`'s window (an attached widget)."""
        self.as_popup(on_dismiss=on_dismiss).show_at(near, x, y)

    def close(self):
        # Item clicks call this (not dismiss_popup), so run on_dismiss here too — e.g. MenuBar
        # title open-highlight.
        self._close_submenu()
        callback = self._take_dismiss()
        if self._popup is not None:
            self._popup.close()
        if callback is not None:
            callback()

    @property
    def is_open(self):
        return bool(self._popup is not None and self._popup.is_open)


def _menu_accelerators(menu):
    """`(shortcut, on_click)` pairs from `menu` and nested submenus."""
    out = []
    for item in menu.items:
        if not isinstance(item, MenuItem) or not item.enabled:
            continue
        if item.shortcut and item.on_click:
            out.append((item.shortcut, item.on_click))
        if item.submenu is not None:
            out.extend(_menu_accelerators(item.submenu))
    return out


class MenuBar:
    """Traditional menu strip: flat titles, hover/open highlight, thin bottom edge.

    Each entry is `(title, Menu)`. Title clicks toggle that menu; accelerators from enabled
    `MenuItem.shortcut`s are registered on the bar when attached.
    """

    def __init__(self, menus):
        self.menus = list(menus)
        self._titles = []  # (Button, Menu) after describe

    def _clear_title_highlights(self):
        for btn, _ in self._titles:
            btn.set_background(None)

    def _toggle(self, title_button, menu):
        theme = get_theme()
        for btn, other in self._titles:
            if other is not menu and other.is_open:
                other.close()
                btn.set_background(None)
        if menu.is_open:
            menu.close()
            title_button.set_background(None)
        else:
            # Clear open highlight if the menu is dismissed by outside click / Escape / item click.
            # Click-through: with this menu open, clicking another title closes it *and* opens that
            # one. Clicking this title again only closes it (a click on a popup's own anchor is
            # spent closing it — see WidgetTree::popup_press).
            menu.show(title_button, on_dismiss=lambda b=title_button: b.set_background(None), click_through=True)
            title_button.set_background(theme.surface_active)

    @property
    def _fastgui_widget(self):
        theme = get_theme()
        buttons = []
        accelerators = []
        self._titles = []
        for title, menu in self.menus:
            if not isinstance(menu, Menu):
                raise TypeError(f"MenuBar menus must be Menu instances, got {type(menu)!r}")
            holder = {"btn": None}

            def make_click(h=holder, m=menu):
                return lambda: self._toggle(h["btn"], m)

            btn = Button(
                title,
                on_click=make_click(),
                font_size="small",
                text_color=theme.text,
                background=None,
                flat=True,
            )
            holder["btn"] = btn
            buttons.append(btn)
            self._titles.append((btn, menu))
            accelerators.extend(_menu_accelerators(menu))
        # Row of titles + 1px bottom border — reads as a menubar, not a button toolbar.
        titles = Box(
            direction="row",
            gap=0.0,
            children=buttons,
            background=theme.surface,
            accelerators=accelerators,
        )
        edge = Box(direction="row", height=1.0, background=theme.border, children=[])
        return Box(
            direction="column",
            gap=0.0,
            children=[titles, edge],
            background=theme.surface,
        )


class Toolbar:
    """Horizontal strip for Buttons/Labels/spacers under a `MenuBar`."""

    def __init__(self, children):
        self.children = list(children)

    @property
    def _fastgui_widget(self):
        theme = get_theme()
        return Box(
            direction="row",
            gap="small",
            padding="small",
            children=self.children,
            background=theme.surface_alt,
        )


class StatusBar:
    """Bottom strip with muted small text. Call `set_text` to update."""

    def __init__(self, text=""):
        self._text = text
        self._label = None

    def set_text(self, text):
        self._text = text
        if self._label is not None:
            try:
                self._label.set_text(text)
            except RuntimeError:
                pass

    @property
    def _fastgui_widget(self):
        theme = get_theme()
        self._label = Label(self._text, font_size="small", color=theme.text_muted)
        return Box(
            direction="row",
            padding="small",
            children=[self._label],
            background=theme.surface,
            flex_grow=0.0,
        )


class Dialog:
    """Thin modal/modeless wrapper around `Popup`. Open with `show(window)` or
    `window.show_popup(dialog)`."""

    def __init__(self, title, content, buttons=None, modal=True, on_dismiss=None):
        self.title = title
        self.content = content
        self.buttons = list(buttons or [])
        self.modal = bool(modal)
        self.on_dismiss = on_dismiss
        self._popup = None

    def as_popup(self):
        theme = get_theme()
        rows = [Label(self.title, font_size="large"), self.content]
        if self.buttons:
            button_row = []
            for label, callback in self.buttons:

                def make_click(cb=callback):
                    def _click():
                        self.close()
                        if cb is not None:
                            cb()

                    return _click

                button_row.append(Button(label, on_click=make_click()))
            rows.append(Box(direction="row", gap="small", children=button_row))
        body = Box(direction="column", gap="medium", padding="medium", children=rows)
        self._popup = Popup(
            body,
            modal=self.modal,
            on_dismiss=self.on_dismiss,
            background=theme.surface_alt,
            border=theme.border,
        )
        return self._popup

    def show(self, window):
        window.show_popup(self.as_popup())

    def close(self):
        if self._popup is not None:
            self._popup.close()

    @property
    def is_open(self):
        return bool(self._popup is not None and self._popup.is_open)


def pick_color(window, initial=None, on_pick=None):
    """Open an in-app modal RGB color picker on `window`.

    rfd has no color dialog, so this is a small `Dialog` with 0–255 spinners.
    Calls `on_pick(rgba)` on OK or `on_pick(None)` on Cancel/dismiss. Returns the
    `Dialog` (non-blocking — native file dialogs are the blocking pickers).
    """
    theme = get_theme()
    r0, g0, b0, a0 = initial if initial is not None else (*theme.accent[:3], 1.0)
    state = {
        "r": max(0.0, min(1.0, float(r0))),
        "g": max(0.0, min(1.0, float(g0))),
        "b": max(0.0, min(1.0, float(b0))),
        "a": max(0.0, min(1.0, float(a0))),
    }

    def _rgb_text():
        return (
            f"RGB ({int(round(state['r'] * 255))}, "
            f"{int(round(state['g'] * 255))}, {int(round(state['b'] * 255))})"
        )

    readout = Label(_rgb_text(), font_size="small", color=theme.text_muted)

    def sync_from_spins(_value=None):
        state["r"] = r_spin.value / 255.0
        state["g"] = g_spin.value / 255.0
        state["b"] = b_spin.value / 255.0
        readout.set_text(_rgb_text())
        swatch.set_background((state["r"], state["g"], state["b"], state["a"]))

    r_spin = SpinBox(
        value=round(state["r"] * 255), min=0, max=255, step=1, decimals=0, width=72.0, on_change=sync_from_spins
    )
    g_spin = SpinBox(
        value=round(state["g"] * 255), min=0, max=255, step=1, decimals=0, width=72.0, on_change=sync_from_spins
    )
    b_spin = SpinBox(
        value=round(state["b"] * 255), min=0, max=255, step=1, decimals=0, width=72.0, on_change=sync_from_spins
    )

    swatch = Box(
        direction="row",
        height=28.0,
        flex_grow=1.0,
        background=(state["r"], state["g"], state["b"], state["a"]),
        children=[],
    )
    content = Box(
        direction="column",
        gap="small",
        children=[
            Box(
                direction="row",
                gap="small",
                children=[
                    Label("R", font_size="small"),
                    r_spin,
                    Label("G", font_size="small"),
                    g_spin,
                    Label("B", font_size="small"),
                    b_spin,
                ],
            ),
            swatch,
            readout,
        ],
    )

    def on_ok():
        sync_from_spins()
        if on_pick is not None:
            on_pick((state["r"], state["g"], state["b"], state["a"]))

    def on_cancel():
        if on_pick is not None:
            on_pick(None)

    dialog = Dialog(
        "Pick color",
        content,
        buttons=[("OK", on_ok), ("Cancel", on_cancel)],
        modal=True,
        on_dismiss=on_cancel,
    )
    dialog.show(window)
    return dialog


class GroupBox:
    """Titled frame around `content` (Label + bordered/padded Box)."""

    def __init__(self, title, content, padding="medium"):
        self.title = title
        self.content = content
        self.padding = padding

    @property
    def _fastgui_widget(self):
        theme = get_theme()
        # 1px border via nested Boxes (outer border color, inner surface inset by 1).
        inner = Box(
            direction="column",
            gap="small",
            padding=self.padding,
            background=theme.surface,
            children=[
                Label(self.title, font_size="small", color=theme.text_muted),
                self.content,
            ],
        )
        return Box(
            direction="column",
            padding=1.0,
            background=theme.border,
            children=[inner],
        )


class CollapsibleSection:
    """Header row toggles body visibility via `Box.set_display`."""

    def __init__(self, title, content, expanded=True):
        self.title = title
        self.content = content
        self._expanded = bool(expanded)
        self._body = None
        self._header = None

    def set_expanded(self, expanded):
        self._expanded = bool(expanded)
        if self._body is not None:
            self._body.set_display(self._expanded)
        if self._header is not None:
            try:
                self._header.set_text(self._header_label())
            except RuntimeError:
                pass

    def _header_label(self):
        mark = "▼" if self._expanded else "▶"
        return f"{mark}  {self.title}"

    def _toggle(self):
        self.set_expanded(not self._expanded)

    @property
    def expanded(self):
        return self._expanded

    @property
    def _fastgui_widget(self):
        theme = get_theme()
        self._header = Button(
            self._header_label(),
            on_click=self._toggle,
            background=theme.surface_alt,
            text_color=theme.text,
        )
        self._body = Box(
            direction="column",
            padding="small",
            children=[self.content],
            visible=self._expanded,
            flex_grow=1.0,
        )
        return Box(
            direction="column",
            gap=2.0,
            children=[self._header, self._body],
            background=theme.surface,
        )


class StackedWidget:
    """Only one child visible at a time; `set_index(i)` toggles `Display::None` on the rest."""

    def __init__(self, children, index=0):
        self.children = list(children)
        if not self.children:
            raise ValueError("StackedWidget needs at least one child")
        self._index = max(0, min(int(index), len(self.children) - 1))
        self._pages = []

    @property
    def index(self):
        return self._index

    def set_index(self, index):
        if not self.children:
            return
        self._index = max(0, min(int(index), len(self.children) - 1))
        for i, page in enumerate(self._pages):
            page.set_display(i == self._index)

    @property
    def _fastgui_widget(self):
        self._pages = [
            Box(
                direction="column",
                children=[child],
                flex_grow=1.0,
                visible=(i == self._index),
            )
            for i, child in enumerate(self.children)
        ]
        return Box(direction="column", children=self._pages, flex_grow=1.0)


class DockArea:
    """A split-tree of titled `Panel`s (each region resizable by dragging its `Splitter`), with
    drag-to-rearrange: grab a `Panel`'s title bar and drop it on another region to move it there
    (dropping on the edges splits that region; dropping in the center merges into a `Tabs` group,
    including dropping onto an *existing* group to add another tab). Drop near the *window's* own
    outer edge instead of on a specific panel to add the dragged panel as a new row/column
    spanning the *entire* dock area, rather than just a strip the size of whatever panel was under
    the cursor. Pass a `Tabs` instead of a `Panel` to `add_panel` to seed a region with several
    tabbed panels sharing one spot up front. To ungroup, grab a tab's own header segment (not just
    its content) and drag it out onto another region — down to one remaining member, the group
    dissolves back into a plain `Panel`.

    Internally keeps its own mutable tree (not a live `Splitter`/`Panel` object graph — those are
    only ever *built* from this tree, fresh, each time `_fastgui_widget` is read) specifically so
    a rearrange can find, remove, and reinsert a node; the real `Splitter`/`Panel` tree Rust
    builds via `Window.set_content` doesn't expose enough back to Python to do that directly.

    A `Window.add_floating_panel` panel is a real OS window: drag its title bar to move it
    (including outside the main window), drag an edge/corner to resize, and if the main window's
    content is a `DockArea`, drop it onto a dock region (or the window's outer edge) to re-dock.
    Tear a docked panel (or tab) *out* by dragging it: a ghost preview follows the mouse; release
    outside the main window to float, or onto a dock target to re-dock. Click the × on a title
    bar or tab segment to close that panel (floating or docked).

    >>> dock = DockArea()
    >>> dock.add_panel(Panel(title="Viewport", content=viewport_widget), region="center")
    >>> dock.add_panel(Panel(title="Controls", content=controls_box), region="right", size=0.25)
    >>> window.set_content(dock)
    """

    def __init__(self) -> None:
        self._root = None  # None | {"kind": "leaf", "widget": Panel|Tabs}
        #                          | {"kind": "split", "direction": str, "ratio": float, "first": node, "second": node}

    def add_panel(self, panel, region: str = "center", size: float = 0.25) -> None:
        if region not in _REGIONS:
            raise ValueError(f"region must be one of {_REGIONS}, got {region!r}")
        if isinstance(panel, Panel):
            panel.set_rearrange_handler(self._on_rearrange)
            panel.set_close_handler(self._on_close)
        elif isinstance(panel, Tabs):
            for member in panel.panels:
                if isinstance(member, Panel):
                    member.set_rearrange_handler(self._on_rearrange)
                    member.set_close_handler(self._on_close)
        leaf = {"kind": "leaf", "widget": panel}

        if self._root is None:
            if region != "center":
                raise ValueError("the first panel added to a DockArea must use region=\"center\"")
            self._root = leaf
            return
        if region == "center":
            raise ValueError("region=\"center\" is only valid for the first panel added")
        if not (0.0 < size < 1.0):
            raise ValueError("size must be between 0.0 and 1.0 (exclusive)")

        direction = "row" if region in ("left", "right") else "column"
        if region in ("right", "bottom"):
            first, second, ratio = self._root, leaf, 1.0 - size
        else:  # left, top
            first, second, ratio = leaf, self._root, size
        self._root = {"kind": "split", "direction": direction, "ratio": ratio, "first": first, "second": second}

    def _on_rearrange(
        self,
        dragged_id: int,
        target_id: int,
        zone: str,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 320.0,
        height: float = 180.0,
    ) -> None:
        """Bound to every `Panel`'s title-bar drag (see `add_panel`); fires when the user drops
        one over another region (or tears it out to float — `zone == "float"`). Rebuilds
        `self._root` and, if this `DockArea` has already been given to a `Window`, asks it to
        re-attach the new structure — nothing else in this architecture gives a composite widget
        a route to trigger that on its own, see `Window.set_content`'s Rust-side doc comment."""
        if zone == "float":
            self._undock_to_floating(dragged_id, x, y, width, height)
            return
        if dragged_id == target_id:
            return
        # A tab dropped on its own group: extraction rebuilds that group as a new `Tabs` (new
        # id) or collapses it to its last `Panel`, so `target_id` won't survive `_extract`.
        # Remember a sibling tab to re-find the group by afterwards.
        own_group_sibling = None
        if target_id != _ROOT_REGION_ID and self._root is not None:
            target_leaf = _find_leaf(self._root, target_id)
            if target_leaf is not None and isinstance(target_leaf["widget"], Tabs):
                member_ids = [p.id for p in target_leaf["widget"].panels]
                if dragged_id in member_ids:
                    if zone == "center":
                        return  # Already in this group.
                    own_group_sibling = next(i for i in member_ids if i != dragged_id)
        from_floating = False
        if self._root is None:
            # Empty dock (every panel was torn out): only a floating re-dock can fill it.
            window = getattr(self, "_fastgui_window", None)
            if window is None:
                return
            panel = window._peek_floating_panel(dragged_id)
            if panel is None:
                return
            extracted = {"kind": "leaf", "widget": panel}
            remainder = None
            from_floating = True
        else:
            remainder, extracted = _extract(self._root, dragged_id)
            if extracted is None:
                # Not in the dock tree — a floating panel can still re-dock here.
                window = getattr(self, "_fastgui_window", None)
                if window is None:
                    return
                panel = window._peek_floating_panel(dragged_id)
                if panel is None:
                    return
                extracted = {"kind": "leaf", "widget": panel}
                remainder = self._root
                from_floating = True
        if remainder is None:
            # Empty dock accepting a floater, or the only docked panel was the drag source and
            # landed back on itself somehow — just place the extracted leaf as the sole content.
            if from_floating:
                getattr(self, "_fastgui_window")._take_floating_panel(dragged_id)
            self._root = extracted
            window = getattr(self, "_fastgui_window", None)
            if window is not None:
                window.set_content(self)
            return

        if target_id == _ROOT_REGION_ID:
            # Dropped near the *window's* outer edge: wrap the whole remaining layout, not just
            # whichever panel happens to be under the cursor — otherwise there'd be no way to
            # add a panel as a new row/column spanning the full width/height, only ever a strip
            # the size of one single existing panel. `zone` here is never "center" (see
            # `update_panel_drag_hover`'s Rust-side doc comment — the outer-edge check only ever
            # picks an edge). Ratio of 0.25 for the newly-placed panel matches `add_panel`'s own
            # default `size` for the same reason: an edge strip should default to *thin*, not an
            # even 50/50 split.
            direction = "row" if zone in ("left", "right") else "column"
            if zone in ("right", "bottom"):
                first, second, ratio = remainder, extracted, 0.75
            else:
                first, second, ratio = extracted, remainder, 0.25
            new_root = {"kind": "split", "direction": direction, "ratio": ratio, "first": first, "second": second}
        else:
            if own_group_sibling is not None:
                target_leaf = _find_leaf_containing(remainder, own_group_sibling)
                if target_leaf is not None:
                    target_id = target_leaf["widget"].id
            target_leaf = _find_leaf(remainder, target_id)
            if target_leaf is None:
                # Target vanished (e.g. it was the dragged panel's own sibling and got collapsed
                # into it during extraction) — bail out without mutating rather than lose a panel.
                return

            if zone == "center":
                new_group = _tabs_from_center_drop(target_leaf["widget"], extracted["widget"])
                if new_group is None:
                    return
                new_root = _replace(remainder, target_id, {"kind": "leaf", "widget": new_group})
            else:
                direction = "row" if zone in ("left", "right") else "column"
                if zone in ("right", "bottom"):
                    first, second = target_leaf, extracted
                else:
                    first, second = extracted, target_leaf
                new_split = {"kind": "split", "direction": direction, "ratio": 0.5, "first": first, "second": second}
                new_root = _replace(remainder, target_id, new_split)

        if from_floating:
            getattr(self, "_fastgui_window")._take_floating_panel(dragged_id)
        self._root = new_root
        window = getattr(self, "_fastgui_window", None)
        if window is not None:
            window.set_content(self)

    def _on_close(self, panel_id: int) -> None:
        """Close a docked or floating panel (title-bar / tab ×). Floating panels are removed from
        the window; docked panels are extracted from the tree (empty dock stays as content)."""
        window = getattr(self, "_fastgui_window", None)
        if window is not None and window._peek_floating_panel(panel_id) is not None:
            window._take_floating_panel(panel_id)
            return
        if self._root is None:
            return
        remainder, extracted = _extract(self._root, panel_id)
        if extracted is None:
            return
        self._root = remainder
        if window is not None:
            window.set_content(self)

    def _undock_to_floating(self, dragged_id: int, x: float, y: float, width: float, height: float) -> None:
        """Tear `dragged_id` out of the dock tree into a real floating OS window."""
        if self._root is None:
            return
        remainder, extracted = _extract(self._root, dragged_id)
        if extracted is None:
            return
        widget = extracted["widget"]
        if not isinstance(widget, Panel):
            return
        self._root = remainder
        window = getattr(self, "_fastgui_window", None)
        if window is None:
            return
        window.set_content(self)
        window.add_floating_panel(widget, x, y, max(width, 160.0), max(height, 100.0))

    def _build(self, node):
        if node["kind"] == "leaf":
            return node["widget"]
        return Splitter(
            first=self._build(node["first"]),
            second=self._build(node["second"]),
            direction=node["direction"],
            ratio=node["ratio"],
        )

    @property
    def _fastgui_widget(self):
        """Consumed by `fastgui-py`'s `describe()` — lets `window.set_content(dock)` work
        without the Rust side needing to know `DockArea` exists at all. Builds a fresh
        `Splitter`/`Panel` tree from `self._root` every time it's read (including on a
        rearrange-triggered rebuild) rather than keeping one around persistently, since
        `self._root` is the actual source of truth and is what rearranges mutate."""
        if self._root is None:
            # Every panel was torn out to float — keep the DockArea as window content so a
            # later re-dock still has a rearrange handler to bind to.
            return Box(direction="column", children=[])
        return self._build(self._root)


def _tabs_from_center_drop(target_widget, dragged_widget):
    """Build the `Tabs` group a center-drop should produce: grow an existing group by appending
    the dragged panel (and selecting it), or form a new two-tab group from two standalone
    `Panel`s. Returns `None` for a target that isn't a drop-into-tabs surface (shouldn't happen
    — regions are only ever `Panel`/`Tabs`)."""
    if isinstance(target_widget, Tabs):
        members = list(target_widget.panels) + [dragged_widget]
        return Tabs(members, active=len(members) - 1)
    if isinstance(target_widget, Panel):
        return Tabs([target_widget, dragged_widget])
    return None


def _extract(node, region_id: int):
    """Remove the leaf whose widget has `.id == region_id` from `node`'s subtree, collapsing its
    parent `split` if that leaf was one side of it. Returns `(subtree_without_it, removed_leaf)`
    — both `None` if `region_id` wasn't found anywhere in `node`.

    Also looks *inside* a `Tabs` leaf for a member `Panel` matching `region_id` — this is how
    dragging a tab back out of a group (ungrouping) is implemented: a `TabBar` tab drag reports
    the dragged-out `Panel`'s own id (see `WidgetKind::TabBar`'s Rust-side doc comment), not the
    `Tabs` group's id, so it's found here rather than at the leaf-`widget`-id check below. Down
    to one remaining member, the `Tabs` collapses back into a plain `Panel` leaf rather than a
    pointless one-tab group."""
    if node is None:
        return None, None
    if node["kind"] == "leaf" and isinstance(node["widget"], Tabs):
        tabs = node["widget"]
        panels = list(tabs.panels)
        for index, panel in enumerate(panels):
            if panel.id == region_id:
                extracted = {"kind": "leaf", "widget": panel}
                remaining = panels[:index] + panels[index + 1:]
                if len(remaining) == 1:
                    return {"kind": "leaf", "widget": remaining[0]}, extracted
                new_active = min(tabs.active, len(remaining) - 1)
                return {"kind": "leaf", "widget": Tabs(remaining, active=new_active)}, extracted
        return node, None
    if node["kind"] == "leaf":
        return (None, node) if node["widget"].id == region_id else (node, None)
    new_first, extracted = _extract(node["first"], region_id)
    if extracted is not None:
        return (node["second"], extracted) if new_first is None else ({**node, "first": new_first}, extracted)
    new_second, extracted = _extract(node["second"], region_id)
    if extracted is not None:
        return (node["first"], extracted) if new_second is None else ({**node, "second": new_second}, extracted)
    return node, None


def _find_leaf(node, region_id: int):
    if node is None:
        return None
    if node["kind"] == "leaf":
        return node if node["widget"].id == region_id else None
    return _find_leaf(node["first"], region_id) or _find_leaf(node["second"], region_id)


def _find_leaf_containing(node, panel_id: int):
    """The leaf holding `panel_id`: the `Panel` leaf itself, or the `Tabs` leaf it's a tab of."""
    if node is None:
        return None
    if node["kind"] == "leaf":
        widget = node["widget"]
        if widget.id == panel_id:
            return node
        if isinstance(widget, Tabs) and any(p.id == panel_id for p in widget.panels):
            return node
        return None
    return _find_leaf_containing(node["first"], panel_id) or _find_leaf_containing(node["second"], panel_id)


def _replace(node, region_id: int, replacement):
    """Swap the leaf whose widget has `.id == region_id` for `replacement` (itself a leaf or a
    split — this is how a drop wraps the target in a new `Splitter`/merges it into a `Tabs`)."""
    if node is None:
        return None
    if node["kind"] == "leaf":
        return replacement if node["widget"].id == region_id else node
    return {**node, "first": _replace(node["first"], region_id, replacement), "second": _replace(node["second"], region_id, replacement)}


__all__ = [
    "Box",
    "Button",
    "Checkbox",
    "CollapsibleSection",
    "ComboBox",
    "CudaSurface",
    "Dialog",
    "DockArea",
    "Grid",
    "GroupBox",
    "Image",
    "Label",
    "ListView",
    "Menu",
    "MenuBar",
    "MenuItem",
    "MenuSeparator",
    "NumericScrub",
    "Panel",
    "Popup",
    "ProgressBar",
    "Radio",
    "ScrollArea",
    "Slider",
    "SpinBox",
    "Splitter",
    "StackedWidget",
    "StatusBar",
    "Tabs",
    "TextArea",
    "TextInput",
    "Theme",
    "Toggle",
    "Toolbar",
    "Viewport",
    "Window",
    "get_theme",
    "open_file_dialog",
    "pick_color",
    "save_file_dialog",
    "set_theme",
]
