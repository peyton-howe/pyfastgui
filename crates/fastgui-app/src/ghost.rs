use fastgui_core::taffy::prelude::*;
use fastgui_core::widget::{Color, WidgetKind, WidgetTree};

/// Minimal title-bar + body tree for the tear-off ghost preview window.
pub fn build_tear_ghost_tree(title: &str) -> WidgetTree {
    let mut tree = WidgetTree::new();
    let root = tree.root();
    tree.set_style(
        root,
        Style {
            flex_direction: FlexDirection::Column,
            size: Size {
                width: Dimension::percent(1.0),
                height: Dimension::percent(1.0),
            },
            ..Default::default()
        },
    );
    let title_id = tree.new_node(
        Style {
            size: Size {
                width: Dimension::percent(1.0),
                height: Dimension::length(28.0),
            },
            flex_grow: 0.0,
            ..Default::default()
        },
        WidgetKind::PanelTitleBar {
            panel_id: 0,
            title: title.to_owned(),
            font_size: 14.0,
            text_color: Color([0.92, 0.93, 0.95, 1.0]),
            background: Color([0.22, 0.30, 0.45, 1.0]),
            on_drop: None,
            on_close: None,
            floating: true,
            container_id: None,
        },
    );
    let body_id = tree.new_node(
        Style {
            size: Size {
                width: Dimension::percent(1.0),
                height: Dimension::auto(),
            },
            flex_grow: 1.0,
            ..Default::default()
        },
        WidgetKind::Container {
            background: Color([0.16, 0.20, 0.28, 1.0]),
            region_id: None,
        },
    );
    tree.add_child(root, title_id);
    tree.add_child(root, body_id);
    tree
}
