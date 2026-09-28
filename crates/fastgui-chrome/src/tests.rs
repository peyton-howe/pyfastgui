use fastgui_core::taffy::prelude::*;
use fastgui_core::widget::WidgetId;

use super::*;

/// Blitting a cached run must draw what the old per-pixel `fill_rect` path did. At a
/// whole-pixel origin that path's anti-aliasing covers exactly one pixel per glyph pixel, so the
/// two should agree to within rounding.
#[test]
fn cached_run_blit_matches_per_pixel_fill_rect() {
    let rect = WidgetRect { x: 3.0, y: 2.0, width: 240.0, height: 30.0 };
    let (text, font_size, color) = ("Hello, chrome × 123 (glyphs)", 16.0, Color([0.9, 0.8, 0.2, 1.0]));
    let background = tiny_skia::Color::from_rgba8(25, 28, 33, 255);

    let mut chrome = ChromeRenderer::new();
    let mut ops = Vec::new();
    chrome.push_text(&mut ops, rect, text, font_size, color);
    let [Op::Text { run, x, y, .. }] = ops.as_slice() else { panic!("expected one text op") };
    let mut blitted = Pixmap::new(260, 40).unwrap();
    blitted.fill(background);
    blit_sprite(&mut blitted, run, x + run.x, y + run.y, None);

    let mut per_pixel = Pixmap::new(260, 40).unwrap();
    per_pixel.fill(background);
    let metrics = Metrics::new(font_size, font_size * 1.25);
    let mut buffer = Buffer::new(&mut chrome.font_system, metrics);
    buffer.set_size(&mut chrome.font_system, Some(rect.width), Some(rect.height));
    buffer.set_text(&mut chrome.font_system, text, &Attrs::new(), Shaping::Advanced);
    buffer.draw(&mut chrome.font_system, &mut chrome.swash_cache, to_cosmic_color(color), |x, y, w, h, c| {
        if c.a() == 0 {
            return;
        }
        let mut paint = Paint::default();
        paint.set_color_rgba8(c.r(), c.g(), c.b(), c.a());
        paint.anti_alias = true;
        if let Some(r) = tiny_skia::Rect::from_xywh(rect.x + x as f32, rect.y + y as f32, w as f32, h as f32) {
            per_pixel.fill_rect(r, &paint, Transform::identity(), None);
        }
    });

    let drawn = blitted.pixels().iter().filter(|p| (p.red(), p.green(), p.blue()) != (25, 28, 33)).count();
    assert!(drawn > 200, "expected visible text, only {drawn} pixels changed");
    let max_diff = blitted.data().iter().zip(per_pixel.data()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(max_diff <= 2, "cached run differs from per-pixel fill_rect by up to {max_diff}");
}

#[test]
fn blend_premultiplied_endpoints() {
    let mut px = [10, 20, 30, 255];
    blend_premultiplied(&mut px, 200, 100, 50, 0);
    assert_eq!(px, [10, 20, 30, 255], "zero alpha leaves the pixel alone");
    blend_premultiplied(&mut px, 200, 100, 50, 255);
    assert_eq!(px, [200, 100, 50, 255], "full alpha replaces it");
    let mut px = [0, 0, 0, 255];
    blend_premultiplied(&mut px, 255, 255, 255, 128);
    assert_eq!(px, [128, 128, 128, 255], "half coverage of white over black");
}

#[test]
fn text_runs_are_reused_until_evicted() {
    let mut chrome = ChromeRenderer::new();
    let rect = WidgetRect { x: 0.0, y: 0.0, width: 100.0, height: 20.0 };
    let color = Color([1.0, 1.0, 1.0, 1.0]);
    let mut first = Vec::new();
    chrome.push_text(&mut first, rect, "same", 14.0, color);
    let mut second = Vec::new();
    chrome.push_text(&mut second, WidgetRect { x: 50.0, ..rect }, "same", 14.0, color);
    let (Op::Text { run: a, .. }, Op::Text { run: b, .. }) = (&first[0], &second[0]) else { panic!() };
    assert!(Arc::ptr_eq(a, b), "same text/size/box/color at a new position reuses the shaped run");

    chrome.text_cache.generation += TEXT_CACHE_KEEP_FRAMES + 1;
    chrome.text_cache.evict();
    assert!(chrome.text_cache.runs.is_empty(), "runs unused for the keep window are evicted");
}

#[test]
fn merge_damage_joins_neighbours_and_falls_back_to_full() {
    let window = PixelRect { x: 0, y: 0, width: 1000, height: 1000 };
    let a = PixelRect { x: 10, y: 10, width: 20, height: 20 };
    let b = PixelRect { x: 25, y: 10, width: 20, height: 20 };
    let far = PixelRect { x: 800, y: 800, width: 10, height: 10 };
    let merged = merge_damage(vec![a, b, far], window).unwrap();
    assert_eq!(merged.len(), 2);
    assert!(merged.contains(&PixelRect { x: 10, y: 10, width: 35, height: 20 }));
    assert!(merged.contains(&far));

    let huge = PixelRect { x: 0, y: 0, width: 1000, height: 600 };
    assert!(merge_damage(vec![huge], window).is_none(), "over half the window repaints in full");
}

struct TestTree {
    tree: WidgetTree,
    slider: WidgetId,
    label: WidgetId,
    button: WidgetId,
    panel_rect_region: u64,
}

fn test_tree() -> TestTree {
    let mut tree = WidgetTree::new();
    let text = Color([0.9, 0.9, 0.92, 1.0]);
    let panel = tree.new_node(
        Style { flex_direction: FlexDirection::Column, flex_grow: 1.0, ..Default::default() },
        WidgetKind::Container { background: Color([0.16, 0.17, 0.2, 1.0]), region_id: Some(7) },
    );
    let root = tree.root();
    tree.add_child(root, panel);
    let title = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: length(28.0_f32) }, ..Default::default() },
        WidgetKind::PanelTitleBar {
            panel_id: 7,
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
    for i in 0..6 {
        let l = tree.new_node(
            Style::default(),
            WidgetKind::Label { text: format!("Row {i}: some text"), font_size: 14.0, color: text },
        );
        tree.add_child(panel, l);
        label.get_or_insert(l);
    }
    let slider = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: length(20.0_f32) }, ..Default::default() },
        WidgetKind::Slider {
            value: 0.25,
            min: 0.0,
            max: 1.0,
            track_color: Color([0.3, 0.3, 0.35, 1.0]),
            thumb_color: Color([0.4, 0.65, 1.0, 1.0]),
            on_change: None,
        },
    );
    tree.add_child(panel, slider);
    let button = tree.new_node(
        Style::default(),
        WidgetKind::Button {
            text: "Apply".into(),
            font_size: 14.0,
            text_color: text,
            background: Color([0.25, 0.3, 0.4, 1.0]),
            on_click: None,
        },
    );
    tree.add_child(panel, button);
    TestTree { tree, slider, label: label.unwrap(), button, panel_rect_region: 7 }
}

/// Rasterize incrementally, then from scratch, and require identical pixels. Returns what the
/// incremental pass reported: `None` = nothing changed, `Some(None)` = full repaint,
/// `Some(Some(rects))` = partial repaint of `rects`.
fn check_matches_full(
    chrome: &mut ChromeRenderer,
    tree: &mut WidgetTree,
    size: (u32, u32),
    drop: Option<(WidgetRect, DropZone)>,
    scale: f32,
) -> Option<Option<Vec<PixelRect>>> {
    let (w, h) = size;
    tree.compute_layout(w as f32 / scale, h as f32 / scale);
    let (incremental, damage) = match chrome.rasterize(tree, w, h, drop, scale) {
        Some(frame) => (frame.data.to_vec(), Some(frame.damage.map(<[PixelRect]>::to_vec))),
        None => (chrome.frame.as_ref().unwrap().data().to_vec(), None),
    };
    chrome.invalidate();
    let full = chrome.rasterize(tree, w, h, drop, scale).expect("full repaint").data.to_vec();
    let mismatched: Vec<_> = incremental
        .as_chunks::<4>()
        .0
        .iter()
        .zip(full.as_chunks::<4>().0)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| ((i as u32 % w, i as u32 / w), a.to_vec(), b.to_vec()))
        .take(5)
        .collect();
    assert!(
        mismatched.is_empty(),
        "scale {scale}: incremental frame differs from a full repaint at (pixel, incremental, full) \
         {mismatched:?}; damage {damage:?}"
    );
    damage
}

#[test]
fn incremental_repaint_matches_full_repaint() {
    for scale in [1.0f32, 1.5, 2.0] {
        let size = ((320.0 * scale) as u32, (260.0 * scale) as u32);
        let TestTree { mut tree, slider, label, button, panel_rect_region } = test_tree();
        let mut chrome = ChromeRenderer::new();

        let first = check_matches_full(&mut chrome, &mut tree, size, None, scale);
        assert!(matches!(first, Some(None)), "first frame is a full repaint");
        assert_eq!(check_matches_full(&mut chrome, &mut tree, size, None, scale), None, "no change, no frame");

        for value in [0.0, 0.5, 0.51, 1.0] {
            tree.mutate_kind(slider, |k| {
                if let WidgetKind::Slider { value: v, .. } = k {
                    *v = value;
                }
            });
            let damage = check_matches_full(&mut chrome, &mut tree, size, None, scale);
            let rects = damage.flatten().expect("a slider move is a partial repaint");
            let area: u64 = rects.iter().map(PixelRect::area).sum();
            assert!(area < u64::from(size.0 * size.1) / 4, "slider damage too large: {rects:?}");
        }

        tree.mutate_kind(label, |k| {
            if let WidgetKind::Label { text, .. } = k {
                *text = "A much longer replacement string that reflows".into();
            }
        });
        check_matches_full(&mut chrome, &mut tree, size, None, scale);

        tree.mutate_kind(button, |k| {
            if let WidgetKind::Button { background, .. } = k {
                *background = Color([0.8, 0.2, 0.2, 1.0]);
            }
        });
        check_matches_full(&mut chrome, &mut tree, size, None, scale);

        tree.compute_layout(size.0 as f32 / scale, size.1 as f32 / scale);
        let region = tree.find_region_rect(panel_rect_region).unwrap();
        for zone in [DropZone::Left, DropZone::Bottom, DropZone::Center] {
            check_matches_full(&mut chrome, &mut tree, size, Some((region, zone)), scale);
        }
        check_matches_full(&mut chrome, &mut tree, size, None, scale);

        tree.set_display(label, false);
        check_matches_full(&mut chrome, &mut tree, size, None, scale);
        tree.set_display(label, true);
        check_matches_full(&mut chrome, &mut tree, size, None, scale);

        let bigger = (size.0 + 10, size.1);
        assert!(matches!(check_matches_full(&mut chrome, &mut tree, bigger, None, scale), Some(None)));
    }
}

/// Draw the tree with the GPU path (quads through `gpu::emulate`, the CPU replica of the
/// shaders, over an atlas built only from uploads so far) and with the CPU painter, and compare.
/// Solid fills and text must agree to rounding; the slider thumb's anti-aliased rim may differ
/// (analytic coverage vs tiny-skia's supersampling), so pixels in circle bounds get more slack.
/// Returns whether `build_quads` emitted a frame.
fn check_gpu_matches_cpu(
    gpu: &mut ChromeRenderer,
    cpu: &mut ChromeRenderer,
    atlas: &mut Vec<u8>,
    tree: &mut WidgetTree,
    size: (u32, u32),
    drop: Option<(WidgetRect, DropZone)>,
    scale: f32,
) -> bool {
    let (w, h) = size;
    tree.compute_layout(w as f32 / scale, h as f32 / scale);
    let emitted = match gpu.build_quads(tree, w, h, drop, scale) {
        Some(frame) => {
            let atlas_size = atlas_side(atlas);
            gpu::apply_uploads(atlas, atlas_size, &frame);
            true
        }
        None => false,
    };
    let atlas_size = atlas_side(atlas);
    let drawn = gpu::emulate(&gpu.gpu.quads, w, h, atlas, atlas_size);
    cpu.invalidate();
    let reference = cpu.rasterize(tree, w, h, drop, scale).expect("full repaint").data.to_vec();

    let circles: Vec<PixelRect> = gpu
        .gpu
        .quads
        .iter()
        .filter(|q| q.kind == fastgui_core::QUAD_CIRCLE || q.kind == fastgui_core::QUAD_CIRCLE_CLIPPED)
        .map(|q| pixel_bounds(q.rect[0], q.rect[1], q.rect[2], q.rect[3], PixelRect { x: 0, y: 0, width: w, height: h }))
        .collect();
    for (i, (a, b)) in drawn.as_chunks::<4>().0.iter().zip(reference.as_chunks::<4>().0).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let px = PixelRect { x, y, width: 1, height: 1 };
        let diff = a.iter().zip(b).map(|(p, q)| p.abs_diff(*q)).max().unwrap();
        let allowed = if circles.iter().any(|c| c.intersects(&px)) { 96 } else { 3 };
        assert!(
            diff <= allowed,
            "scale {scale}: GPU path differs from CPU painter by {diff} at ({x}, {y}): {a:?} vs {b:?}"
        );
    }
    emitted
}

fn atlas_side(atlas: &[u8]) -> u32 {
    ((atlas.len() / 4) as f64).sqrt() as u32
}

#[test]
fn gpu_quads_match_cpu_painter() {
    for scale in [1.0f32, 1.5, 2.0] {
        let size = ((320.0 * scale) as u32, (260.0 * scale) as u32);
        let TestTree { mut tree, slider, label, button, panel_rect_region } = test_tree();
        let (mut gpu, mut cpu, mut atlas) = (ChromeRenderer::new(), ChromeRenderer::new(), Vec::new());
        let mut check = |tree: &mut WidgetTree, drop| check_gpu_matches_cpu(&mut gpu, &mut cpu, &mut atlas, tree, size, drop, scale);

        assert!(check(&mut tree, None), "first frame is emitted");
        assert!(!check(&mut tree, None), "no change, no frame");

        for value in [0.0, 0.5, 0.51, 1.0] {
            tree.mutate_kind(slider, |k| {
                if let WidgetKind::Slider { value: v, .. } = k {
                    *v = value;
                }
            });
            assert!(check(&mut tree, None));
        }
        tree.mutate_kind(label, |k| {
            if let WidgetKind::Label { text, .. } = k {
                *text = "A much longer replacement string that reflows".into();
            }
        });
        check(&mut tree, None);
        tree.mutate_kind(button, |k| {
            if let WidgetKind::Button { background, .. } = k {
                *background = Color([0.8, 0.2, 0.2, 1.0]);
            }
        });
        check(&mut tree, None);
        tree.compute_layout(size.0 as f32 / scale, size.1 as f32 / scale);
        let region = tree.find_region_rect(panel_rect_region).unwrap();
        for zone in [DropZone::Left, DropZone::Bottom, DropZone::Center] {
            check(&mut tree, Some((region, zone)));
        }
        tree.set_display(label, false);
        check(&mut tree, None);
    }
}

/// A label whose text changes every frame fills the atlas with runs nobody draws any more; the
/// atlas must repack (and grow when needed) without ever drawing a stale or missing run.
#[test]
fn gpu_atlas_repacks_under_text_churn() {
    let size = (640, 520);
    let TestTree { mut tree, label, .. } = test_tree();
    let (mut gpu, mut cpu, mut atlas) = (ChromeRenderer::new(), ChromeRenderer::new(), Vec::new());
    let mut repacks = 0;
    for i in 0..400 {
        tree.mutate_kind(label, |k| {
            if let WidgetKind::Label { text, .. } = k {
                *text = format!("Churning label number {i} with enough text to take room");
            }
        });
        tree.compute_layout(size.0 as f32 / 2.0, size.1 as f32 / 2.0);
        let frame = gpu.build_quads(&tree, size.0, size.1, None, 2.0).expect("text changed");
        repacks += usize::from(frame.atlas_repacked);
        let side = atlas_side(&atlas);
        gpu::apply_uploads(&mut atlas, side, &frame);
        if i % 50 == 49 {
            check_gpu_matches_cpu(&mut gpu, &mut cpu, &mut atlas, &mut tree, size, None, 2.0);
        }
    }
    assert!(repacks > 0, "400 distinct runs should overflow a {}² atlas at least once", atlas_side(&atlas));
}

/// A tree with no text still gets a real atlas, so backends never create a 0×0 texture.
#[test]
fn gpu_atlas_is_never_zero_sized() {
    let mut tree = WidgetTree::new();
    let panel = tree.new_node(
        Style { flex_grow: 1.0, ..Default::default() },
        WidgetKind::Container { background: Color([0.16, 0.17, 0.2, 1.0]), region_id: None },
    );
    let root = tree.root();
    tree.add_child(root, panel);
    tree.compute_layout(320.0, 200.0);
    let mut gpu = ChromeRenderer::new();
    let frame = gpu.build_quads(&tree, 320, 200, None, 1.0).expect("first frame");
    assert!(frame.atlas_size > 0, "text-free first frame produced a {0}×{0} atlas", frame.atlas_size);
    assert!(frame.atlas_uploads.is_empty());
}

/// A `Display::None` slider (inactive tab content) must not leave its thumb at the origin —
/// that was the stray blue quarter-circle in `tabs_demo`'s top-left corner.
#[test]
fn hidden_slider_does_not_draw_thumb_at_origin() {
    let mut tree = WidgetTree::new();
    let root = tree.root();
    let visible = tree.new_node(
        Style { flex_grow: 1.0, ..Default::default() },
        WidgetKind::Container { background: Color([0.16, 0.17, 0.2, 1.0]), region_id: None },
    );
    tree.add_child(root, visible);
    let hidden = tree.new_node(
        Style { flex_grow: 1.0, display: Display::None, ..Default::default() },
        WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
    );
    tree.add_child(root, hidden);
    let slider = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: length(24.0_f32) }, ..Default::default() },
        WidgetKind::Slider {
            value: 0.3,
            min: 0.0,
            max: 1.0,
            track_color: Color([0.3, 0.3, 0.35, 1.0]),
            thumb_color: Color([0.4, 0.65, 1.0, 1.0]),
            on_change: None,
        },
    );
    tree.add_child(hidden, slider);
    tree.compute_layout(400.0, 300.0);

    let mut gpu = ChromeRenderer::new();
    let frame = gpu.build_quads(&tree, 400, 300, None, 1.0).expect("first frame");
    let near_origin = frame.quads.iter().filter(|q| {
        q.kind == fastgui_core::QUAD_CIRCLE && q.rect[0] < 8.0 && q.rect[1] < 8.0
    });
    assert_eq!(near_origin.count(), 0, "hidden slider must not emit a circle near (0,0)");

    let mut cpu = ChromeRenderer::new();
    let painted = cpu.rasterize(&tree, 400, 300, None, 1.0).expect("first frame");
    // Background is ~[0.10, 0.11, 0.13]; the thumb is bright blue — any corner pixel that far
    // from the clear color would be the bug.
    let corner = &painted.data[..4];
    assert!(
        corner[2] < 80,
        "top-left pixel looks like the slider thumb: {corner:?}"
    );
}

#[test]
fn focus_ring_adds_four_quads_and_rebuilds() {
    let mut tree = WidgetTree::new();
    let root = tree.root();
    let button = tree.new_node(
        Style { size: Size { width: Dimension::length(80.0), height: Dimension::length(30.0) }, ..Default::default() },
        WidgetKind::Button {
            text: "Go".into(),
            font_size: 14.0,
            text_color: Color([1.0; 4]),
            background: Color([0.2, 0.2, 0.2, 1.0]),
            on_click: None,
        },
    );
    tree.add_child(root, button);
    tree.compute_layout(200.0, 100.0);

    let mut chrome = ChromeRenderer::new();
    let unfocused = chrome.build_quads(&tree, 200, 100, None, 1.0).expect("first frame").quads.len();
    assert!(chrome.build_quads(&tree, 200, 100, None, 1.0).is_none(), "no change, no rebuild");
    tree.set_focus(Some(button));
    let focused = chrome.build_quads(&tree, 200, 100, None, 1.0).expect("focus change rebuilds").quads.len();
    assert_eq!(focused, unfocused + 4);
}

#[test]
fn caret_stops_are_monotonic_and_round_trip() {
    let mut chrome = ChromeRenderer::new();
    let text = "Hello wörld 👍🏽!";
    let stops = chrome.line_stops(text, 16.0);
    assert_eq!(stops.last().map(|s| s.0), Some(text.len()));
    assert!(stops.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1), "{stops:?}");
    for &(index, x) in stops.iter() {
        assert_eq!(chrome.index_at(text, 16.0, x + 0.2), index);
        assert_eq!(chrome.caret_x(text, 16.0, index), x);
    }
    assert_eq!(chrome.index_at(text, 16.0, -50.0), 0);
    assert_eq!(chrome.index_at(text, 16.0, 1e6), text.len());
    assert_eq!(chrome.caret_x("", 16.0, 0), 0.0);
}

fn text_input_tree(text: &str, width: f32, scroll: f32) -> (WidgetTree, WidgetId) {
    let mut tree = WidgetTree::new();
    let root = tree.root();
    let input = tree.new_node(
        Style { size: Size { width: Dimension::length(width), height: Dimension::length(32.0) }, ..Default::default() },
        WidgetKind::TextInput {
            edit: fastgui_core::text_edit::TextEdit::new(text),
            placeholder: "type here".into(),
            font_size: 16.0,
            text_color: Color([1.0; 4]),
            placeholder_color: Color([0.5, 0.5, 0.5, 1.0]),
            background: Color([0.2, 0.2, 0.2, 1.0]),
            selection_color: Color([0.2, 0.4, 0.8, 1.0]),
            scroll,
            preedit: None,
            on_change: None,
            on_submit: None,
            mirror: None,
        },
    );
    tree.add_child(root, input);
    tree.compute_layout(400.0, 100.0);
    (tree, input)
}

#[test]
fn long_text_input_line_is_clipped_to_its_field_at_any_scroll() {
    let text = "a long line of text that is much wider than the little field it sits in";
    for scroll in [0.0, 120.0] {
        let (mut tree, input) = text_input_tree(text, 100.0, scroll);
        tree.set_focus(Some(input));
        let mut chrome = ChromeRenderer::new();
        let items = chrome.build_items(&tree, None, 2.0, PixelRect { x: 0, y: 0, width: 800, height: 200 });
        let field = &items.iter().find(|item| item.key == u64::from(input)).expect("field item").ops;
        let text_ops: Vec<_> = field.iter().filter_map(|op| match op { Op::Text { run, x, .. } => Some((run, *x)), _ => None }).collect();
        assert_eq!(text_ops.len(), 1, "one line of text at scroll {scroll}");
        let (run, x) = text_ops[0];
        let padding = (TEXT_INPUT_PADDING * 2.0) as i32;
        assert!(x + run.x >= padding && x + run.x + run.width as i32 <= 200 - padding, "scroll {scroll}: text spills out");
        // Background, focus ring (4), text, caret — no selection.
        assert_eq!(field.len(), 7, "scroll {scroll}");
    }
}

#[test]
fn empty_unfocused_text_input_shows_placeholder_without_caret() {
    let (tree, input) = text_input_tree("", 200.0, 0.0);
    let mut chrome = ChromeRenderer::new();
    let items = chrome.build_items(&tree, None, 1.0, PixelRect { x: 0, y: 0, width: 400, height: 100 });
    let field = &items.iter().find(|item| item.key == u64::from(input)).unwrap().ops;
    assert!(matches!(field.as_slice(), [Op::Fill { .. }, Op::Text { .. }]), "background + placeholder only");
}

/// A 200×90 scroll area at (10, 10) over a column taller than it: labels, a button and a slider
/// whose thumb the clip edge cuts through at some offsets.
fn scroll_scene() -> (WidgetTree, WidgetId) {
    let mut tree = WidgetTree::new();
    let root = tree.root();
    let area = tree.new_node(
        Style {
            flex_direction: FlexDirection::Column,
            size: Size { width: Dimension::length(200.0), height: Dimension::length(90.0) },
            margin: fastgui_core::taffy::prelude::Rect {
                left: LengthPercentageAuto::length(10.0),
                top: LengthPercentageAuto::length(10.0),
                right: LengthPercentageAuto::length(0.0),
                bottom: LengthPercentageAuto::length(0.0),
            },
            ..Default::default()
        },
        WidgetKind::ScrollArea {
            offset: (0.0, 0.0),
            background: Color([0.15, 0.16, 0.2, 1.0]),
            bar_color: Color([1.0, 1.0, 1.0, 0.35]),
        },
    );
    let column = tree.new_node(
        Style { flex_direction: FlexDirection::Column, gap: Size { width: LengthPercentage::length(6.0), height: LengthPercentage::length(6.0) }, ..Default::default() },
        WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
    );
    tree.add_child(root, area);
    tree.add_child(area, column);
    for i in 0..4 {
        let label = tree.new_node(
            Style::default(),
            WidgetKind::Label { text: format!("Row {i} — clipped text"), font_size: 15.0, color: Color([0.9, 0.9, 0.95, 1.0]) },
        );
        tree.add_child(column, label);
    }
    let button = tree.new_node(
        Style { size: Size { width: Dimension::length(120.0), height: Dimension::length(28.0) }, ..Default::default() },
        WidgetKind::Button {
            text: "Button".into(),
            font_size: 14.0,
            text_color: Color([1.0; 4]),
            background: Color([0.3, 0.4, 0.7, 1.0]),
            on_click: None,
        },
    );
    let slider = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: Dimension::length(24.0) }, ..Default::default() },
        WidgetKind::Slider {
            value: 0.3,
            min: 0.0,
            max: 1.0,
            track_color: Color([0.3, 0.3, 0.35, 1.0]),
            thumb_color: Color([0.4, 0.7, 1.0, 1.0]),
            on_change: None,
        },
    );
    tree.add_child(column, button);
    tree.add_child(column, slider);
    tree.set_scroll_container(area);
    tree.compute_layout(240.0, 130.0);
    (tree, area)
}

#[test]
fn clipped_scroll_content_matches_cpu_painter_and_stays_inside() {
    for scale in [1.0f32, 1.5, 2.0] {
        let size = ((240.0 * scale) as u32, (130.0 * scale) as u32);
        let (mut tree, area) = scroll_scene();
        let max = tree.scroll_extent(area).unwrap().1;
        assert!(max > 30.0, "the column overflows: {max}");
        let (mut gpu, mut cpu, mut atlas) = (ChromeRenderer::new(), ChromeRenderer::new(), Vec::new());
        for offset in [0.0, 7.5, 13.25, 41.0, max] {
            tree.set_scroll_offset(area, 0.0, offset);
            check_gpu_matches_cpu(&mut gpu, &mut cpu, &mut atlas, &mut tree, size, None, scale);
            let clip = Clip::from_rect(scale_rect(tree.absolute_rect(area).unwrap(), scale));
            // Everything but the window background is inside the area.
            for quad in &gpu.gpu.quads[1..] {
                let [l, t, r, b] = quad.rect;
                let inside = l >= clip.left as f32 && t >= clip.top as f32 && r <= clip.right as f32 && b <= clip.bottom as f32;
                assert!(inside, "scale {scale} offset {offset}: quad {quad:?} leaves the scroll area {clip:?}");
            }
        }
        assert!(
            gpu.gpu.quads.iter().any(|q| q.kind == fastgui_core::QUAD_SPRITE),
            "some text is still drawn at the end"
        );
    }
}

#[test]
fn clip_ops_cuts_fills_and_tags_straddling_sprites() {
    let mut chrome = ChromeRenderer::new();
    let mut ops = Vec::new();
    push_fill(&mut ops, WidgetRect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 }, Color([1.0; 4]));
    push_fill(&mut ops, WidgetRect { x: 200.0, y: 0.0, width: 10.0, height: 10.0 }, Color([1.0; 4]));
    chrome.push_text(&mut ops, WidgetRect { x: 30.0, y: 40.0, width: 200.0, height: 30.0 }, "straddles", 16.0, Color([1.0; 4]));
    chrome.push_text(&mut ops, WidgetRect { x: 12.0, y: 12.0, width: 200.0, height: 30.0 }, "in", 16.0, Color([1.0; 4]));
    let clip = Clip { left: 10, top: 10, right: 60, bottom: 50 };
    let ops = clip_ops(ops, clip);
    assert_eq!(ops.len(), 3, "the fill outside the clip is dropped");
    assert!(matches!(ops[0], Op::Fill { left: 10.0, top: 10.0, right: 60.0, bottom: 50.0, .. }));
    assert!(matches!(ops[1], Op::Text { clip: Some(c), .. } if c == clip), "partly outside: carries the clip");
    assert!(matches!(ops[2], Op::Text { clip: None, .. }), "wholly inside: no clip needed");
}

#[test]
fn popup_quads_form_the_overlay_range() {
    use fastgui_core::widget::PopupAnchor;
    let (mut tree, area) = scroll_scene();
    let mut chrome = ChromeRenderer::new();
    let base = chrome.build_quads(&tree, 240, 130, None, 1.0).expect("first frame");
    assert_eq!(base.overlay_start, base.quads.len(), "no popup: nothing in the overlay range");
    let base_len = base.quads.len();

    let popup_kind = |modal| WidgetKind::Popup {
        anchor: PopupAnchor::Center,
        modal,
        background: Color([0.14, 0.15, 0.18, 1.0]),
        border: Color([0.3, 0.3, 0.4, 1.0]),
        on_dismiss: None,
        restore_focus: None,
        open: None,
    };
    let popup = tree.open_popup(popup_kind(true), |tree, popup| {
        let label = tree.new_node(
            Style::default(),
            WidgetKind::Label { text: "In a popup".into(), font_size: 14.0, color: Color([1.0; 4]) },
        );
        tree.add_child(popup, label);
    });
    tree.compute_layout(240.0, 130.0);
    let frame = chrome.build_quads(&tree, 240, 130, None, 1.0).expect("popup opened");
    let overlay = &frame.quads[frame.overlay_start..];
    assert_eq!(frame.overlay_start, base_len, "the base content is unchanged and comes first");
    assert_eq!(overlay[0].rect, [0.0, 0.0, 240.0, 130.0], "a modal popup dims the whole window first");
    assert!(overlay.iter().any(|q| q.kind == fastgui_core::QUAD_SPRITE), "its text is in the overlay");
    let r = tree.absolute_rect(popup).unwrap();
    assert!((r.x + r.width / 2.0 - 120.0).abs() < 0.5 && (r.y + r.height / 2.0 - 65.0).abs() < 0.5, "centered");

    tree.close_popup(popup);
    tree.compute_layout(240.0, 130.0);
    let _ = area;
    let closed = chrome.build_quads(&tree, 240, 130, None, 1.0).expect("popup closed");
    assert_eq!((closed.quads.len(), closed.overlay_start), (base_len, base_len));
}
