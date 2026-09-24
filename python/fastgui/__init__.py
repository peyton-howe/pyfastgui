from ._fastgui import Box, Button, CudaSurface, Label, Panel, Slider, Splitter, Tabs, Viewport, Window

_REGIONS = ("center", "left", "right", "top", "bottom")

# Matches `fastgui-app`'s `ROOT_REGION_ID` — sent as `target_id` when a panel is
# dropped near the *window's* outer edge (not a specific panel's), meaning "wrap the whole
# DockArea", not "split just whatever panel happens to be under the cursor". Never a real
# `Panel`/`Tabs` id (`next_region_id()`, Rust-side, starts at 1).
_ROOT_REGION_ID = 0


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
    "CudaSurface",
    "DockArea",
    "Label",
    "Panel",
    "Slider",
    "Splitter",
    "Tabs",
    "Viewport",
    "Window",
]
