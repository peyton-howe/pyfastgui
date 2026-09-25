//! Shared GPU-backend check (feature `testing`, for backend tests only): run one scenario of
//! chrome changes through `build_quads`, let the backend draw each frame with its real shaders,
//! and compare the pixels it reads back with the CPU painter.

use std::sync::Arc;

use fastgui_core::taffy::prelude::*;
use fastgui_core::widget::{Color, DropZone, WidgetKind, WidgetTree};
use fastgui_core::{ChromeQuads, QUAD_CIRCLE};

use crate::ChromeRenderer;

/// A panel with a closeable title bar, labels and a slider — every quad kind, plus translucency
/// once the drop indicator shows.
fn scene() -> (WidgetTree, fastgui_core::widget::WidgetId, fastgui_core::widget::WidgetId) {
    let mut tree = WidgetTree::new();
    let text = Color([0.9, 0.9, 0.92, 1.0]);
    let panel = tree.new_node(
        Style { flex_direction: FlexDirection::Column, flex_grow: 1.0, gap: length(3.0_f32), ..Default::default() },
        WidgetKind::Container { background: Color([0.16, 0.17, 0.2, 1.0]), region_id: Some(1) },
    );
    let root = tree.root();
    tree.add_child(root, panel);
    let title = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: length(28.0_f32) }, ..Default::default() },
        WidgetKind::PanelTitleBar {
            panel_id: 1,
            title: "Panel".into(),
            font_size: 14.0,
            text_color: text,
            background: Color([0.2, 0.22, 0.26, 1.0]),
            on_drop: None,
            on_close: Some(Arc::new(|_| {})),
            floating: false,
            container_id: None,
        },
    );
    tree.add_child(panel, title);
    let mut label = None;
    for i in 0..5 {
        let l = tree.new_node(
            Style::default(),
            WidgetKind::Label { text: format!("Row {i}: text on the GPU"), font_size: 13.0, color: text },
        );
        tree.add_child(panel, l);
        label.get_or_insert(l);
    }
    let slider = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: length(20.0_f32) }, ..Default::default() },
        WidgetKind::Slider {
            value: 0.3,
            min: 0.0,
            max: 1.0,
            track_color: Color([0.3, 0.3, 0.35, 1.0]),
            thumb_color: Color([0.4, 0.65, 1.0, 1.0]),
            on_change: None,
        },
    );
    tree.add_child(panel, slider);
    (tree, slider, label.expect("five labels"))
}

/// GPU pixels vs the CPU painter: solid fills and text within rounding, the slider thumb's
/// anti-aliased rim within a looser bound (analytic vs supersampled coverage).
fn assert_matches(gpu: &[u8], cpu: &[u8], width: u32, thumb: Option<[f32; 4]>, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: readback size");
    let mut near_thumb_max = 0;
    let mut drawn = 0;
    for (i, (a, b)) in gpu.as_chunks::<4>().0.iter().zip(cpu.as_chunks::<4>().0).enumerate() {
        let (x, y) = ((i as u32 % width) as f32, (i as u32 / width) as f32);
        let near_thumb =
            thumb.is_some_and(|[l, t, r, b]| x >= l - 2.0 && x <= r + 2.0 && y >= t - 2.0 && y <= b + 2.0);
        let diff = a.iter().zip(b).map(|(p, q)| p.abs_diff(*q)).max().expect("4 channels");
        drawn += usize::from(a[3] != 0);
        if near_thumb {
            near_thumb_max = near_thumb_max.max(diff);
            continue;
        }
        assert!(diff <= 3, "{what}: GPU differs from CPU by {diff} at ({x}, {y}): {a:?} vs {b:?}");
    }
    assert_eq!(drawn, gpu.len() / 4, "{what}: the GPU left pixels undrawn");
    assert!(near_thumb_max <= 96, "{what}: slider thumb rim differs by {near_thumb_max}");
}

/// Drive a GPU backend through the scenario at 1× and 2×. `make(width, height)` sets up an
/// offscreen target; `draw(target, frame)` applies `frame` (if `Some`) on top of what it already
/// holds, draws, and returns the target's pixels as RGBA8, top row first.
pub fn check_gpu_backend<G>(
    mut make: impl FnMut(u32, u32) -> G,
    mut draw: impl FnMut(&mut G, Option<ChromeQuads<'_>>) -> Vec<u8>,
) {
    for scale in [1.0f32, 2.0] {
        let (w, h) = ((300.0 * scale) as u32, (220.0 * scale) as u32);
        let (mut tree, slider, label) = scene();
        let (mut quads, mut cpu) = (ChromeRenderer::new(), ChromeRenderer::new());
        let mut target = make(w, h);
        // From the latest emitted frame: an unchanged frame emits nothing, but the thumb is still there.
        let mut thumb = None;

        let mut check = |tree: &mut WidgetTree, drop: Option<(fastgui_core::widget::Rect, DropZone)>, what: &str| {
            tree.compute_layout(w as f32 / scale, h as f32 / scale);
            let frame = quads.build_quads(tree, w, h, drop, scale);
            if let Some(frame) = &frame {
                thumb = frame.quads.iter().find(|q| q.kind == QUAD_CIRCLE).map(|q| q.rect);
            }
            let drawn = draw(&mut target, frame);
            cpu.invalidate();
            let reference = cpu.rasterize(tree, w, h, drop, scale).expect("full repaint").data.to_vec();
            assert_matches(&drawn, &reference, w, thumb, &format!("scale {scale}, {what}"));
        };

        check(&mut tree, None, "first frame");
        tree.mutate_kind(slider, |k| {
            if let WidgetKind::Slider { value, .. } = k {
                *value = 0.77;
            }
        });
        check(&mut tree, None, "slider moved");
        tree.mutate_kind(label, |k| {
            if let WidgetKind::Label { text, .. } = k {
                *text = "Changed text uploaded into the existing atlas".into();
            }
        });
        check(&mut tree, None, "label changed (incremental atlas upload)");
        tree.compute_layout(w as f32 / scale, h as f32 / scale);
        let region = tree.find_region_rect(1).expect("the panel is region 1");
        check(&mut tree, Some((region, DropZone::Right)), "translucent drop indicator");
        check(&mut tree, Some((region, DropZone::Right)), "unchanged frame (no new quads)");
    }
}
