//! Chrome timings for two trees: a dock-demo-sized one (6 panels, ~84 text items) and a
//! 2,000-cell table (50 rows × 40 flex-width columns), the size where per-string costs
//! stop being small.
//!
//! `cargo run --release -p fastgui-chrome --example chrome_bench` (or `scripts/chrome-timing.sh`
//! / `scripts/chrome-timing.ps1`, which also record the CPU and save the output). CPU-only: no
//! GPU, window, or Vulkan SDK needed, so it runs on any machine with a Rust toolchain.
//!
//! Per frame: `layout` is `WidgetTree::compute_layout` (taffy), `raster` is
//! `ChromeRenderer::rasterize`, `copy` is the CPU cost of writing the frame's damage into a
//! texture-sized buffer — what the Vulkan backend's mapped-memory upload does, and a close
//! stand-in for Metal's `replaceRegion` on shared storage (GPU-side sync is not included).
//! `upload` is the bytes copied. `total` is compared with 60 Hz / 120 Hz frame budgets.
//!
//! Dock scene:
//! - `slider drag`: one slider value + its readout label change per frame — the common
//!   "one widget moved" case.
//! - `resize`: window width alternates by 1pt each frame, so every rect moves and the whole
//!   frame repaints (both widths' text stays cached).
//!
//! Table scene:
//! - `1 cell ticks` / `50 cells tick`: cells get a never-before-seen value each frame (worst
//!   case for the text cache — every changed cell is shaped).
//! - `scroll`: a spacer above the table changes height by 1pt each frame, moving every cell
//!   (full repaint; cell text is cached because position isn't part of the key).
//! - `resize warm`: width alternates by 1pt (two sets of cached cell widths).
//! - `resize drag`: width shrinks 1pt every frame. taffy rounds layout to whole pixels, so only
//!   the cells whose rounded width changed (~75 of 2,000) miss the text cache and are reshaped.

use std::hint::black_box;
use std::time::Instant;

use fastgui_chrome::ChromeRenderer;
use fastgui_core::taffy::prelude::*;
use fastgui_core::widget::{Color, WidgetId, WidgetKind, WidgetTree};
use fastgui_core::{ChromeFrame, PixelRect};

const PANELS: usize = 6;
const LABELS_PER_PANEL: usize = 12;

struct Demo {
    tree: WidgetTree,
    slider: WidgetId,
    readout: WidgetId,
}

fn build() -> Demo {
    let mut tree = WidgetTree::new();
    let text = Color([0.9, 0.9, 0.92, 1.0]);
    let row = tree.new_node(
        Style { flex_direction: FlexDirection::Row, flex_grow: 1.0, ..Default::default() },
        WidgetKind::Container { background: Color([0.12, 0.13, 0.15, 1.0]), region_id: None },
    );
    let root = tree.root();
    tree.add_child(root, row);
    let mut slider = None;
    let mut readout = None;
    for column_index in 0..PANELS / 2 {
        let column = tree.new_node(
            Style { flex_direction: FlexDirection::Column, flex_grow: 1.0, ..Default::default() },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
        );
        tree.add_child(row, column);
        for panel_index in 0..2 {
            let id = (column_index * 2 + panel_index) as u64;
            let panel = tree.new_node(
                Style { flex_direction: FlexDirection::Column, flex_grow: 1.0, ..Default::default() },
                WidgetKind::Container { background: Color([0.16, 0.17, 0.2, 1.0]), region_id: Some(id) },
            );
            tree.add_child(column, panel);
            let title = tree.new_node(
                Style { size: Size { width: Dimension::auto(), height: length(28.0_f32) }, ..Default::default() },
                WidgetKind::PanelTitleBar {
                    panel_id: id,
                    title: format!("Panel {id}"),
                    font_size: 14.0,
                    text_color: text,
                    background: Color([0.2, 0.22, 0.26, 1.0]),
                    on_drop: None,
                    on_close: None,
                    floating: false,
                    container_id: None,
                },
            );
            tree.add_child(panel, title);
            for label_index in 0..LABELS_PER_PANEL {
                let label = tree.new_node(
                    Style::default(),
                    WidgetKind::Label { text: format!("Property {label_index}: value {id}.{label_index}"), font_size: 14.0, color: text },
                );
                tree.add_child(panel, label);
                if readout.is_none() {
                    readout = Some(label);
                }
            }
            let s = tree.new_node(
                Style { size: Size { width: Dimension::auto(), height: length(20.0_f32) }, ..Default::default() },
                WidgetKind::Slider {
                    value: 0.5,
                    min: 0.0,
                    max: 1.0,
                    track_color: Color([0.3, 0.3, 0.35, 1.0]),
                    thumb_color: Color([0.4, 0.65, 1.0, 1.0]),
                    on_change: None,
                },
            );
            tree.add_child(panel, s);
            slider.get_or_insert(s);
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
        }
    }
    Demo { tree, slider: slider.unwrap(), readout: readout.unwrap() }
}

struct Table {
    tree: WidgetTree,
    spacer: WidgetId,
    cells: Vec<WidgetId>,
}

const TABLE_ROWS: usize = 50;
const TABLE_COLS: usize = 40;

fn build_table() -> Table {
    let mut tree = WidgetTree::new();
    let text = Color([0.9, 0.9, 0.92, 1.0]);
    let root = tree.root();
    // `flex_shrink: 0` everywhere: the rows overflow the window (like a scrolled table) rather
    // than squeezing the spacer back to zero.
    let spacer = tree.new_node(
        Style { size: Size { width: Dimension::auto(), height: length(0.0_f32) }, flex_shrink: 0.0, ..Default::default() },
        WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
    );
    tree.add_child(root, spacer);
    let mut cells = Vec::new();
    for row_index in 0..TABLE_ROWS {
        let background = if row_index % 2 == 0 { Color([0.14, 0.15, 0.17, 1.0]) } else { Color([0.12, 0.13, 0.15, 1.0]) };
        let row = tree.new_node(
            Style {
                flex_direction: FlexDirection::Row,
                size: Size { width: Dimension::auto(), height: length(20.0_f32) },
                flex_shrink: 0.0,
                ..Default::default()
            },
            WidgetKind::Container { background, region_id: None },
        );
        tree.add_child(root, row);
        for col_index in 0..TABLE_COLS {
            // Flex columns (`flex_basis: 0`) so a window resize changes every cell's width —
            // the realistic worst case for a width-keyed text cache.
            let cell = tree.new_node(
                Style {
                    flex_grow: 1.0,
                    flex_basis: length(0.0_f32),
                    min_size: Size { width: length(0.0_f32), height: Dimension::auto() },
                    ..Default::default()
                },
                WidgetKind::Label { text: format!("{}.{:02}", row_index * 7 + col_index, col_index), font_size: 11.0, color: text },
            );
            tree.add_child(row, cell);
            cells.push(cell);
        }
    }
    Table { tree, spacer, cells }
}

fn set_label(tree: &mut WidgetTree, id: WidgetId, value: String) {
    tree.mutate_kind(id, |kind| {
        if let WidgetKind::Label { text, .. } = kind {
            *text = value;
        }
    });
}

/// Write the frame's damage (or all of it) into `texture`, the way the Vulkan backend writes
/// into its mapped image. Returns bytes copied.
fn copy_to_texture(frame: Option<ChromeFrame<'_>>, texture: &mut Vec<u8>) -> u64 {
    let Some(frame) = frame else { return 0 };
    if texture.len() != frame.data.len() {
        *texture = vec![0; frame.data.len()];
    }
    let stride = frame.width as usize * 4;
    let full = [PixelRect { x: 0, y: 0, width: frame.width, height: frame.height }];
    let rects = frame.damage.unwrap_or(&full);
    let mut bytes = 0;
    for rect in rects {
        let (x, row_bytes) = (rect.x as usize * 4, rect.width as usize * 4);
        for y in rect.y as usize..rect.bottom() as usize {
            let start = y * stride + x;
            texture[start..start + row_bytes].copy_from_slice(&frame.data[start..start + row_bytes]);
        }
        bytes += rect.area() * 4;
    }
    black_box(&texture);
    bytes
}

#[derive(Default)]
struct Sample {
    layout: f64,
    raster: f64,
    copy: f64,
    bytes: u64,
}

/// A window's chrome state: the renderer plus a stand-in for its GPU texture.
struct Target {
    chrome: ChromeRenderer,
    texture: Vec<u8>,
}

impl Target {
    fn new() -> Self {
        Self { chrome: ChromeRenderer::new(), texture: Vec::new() }
    }
}

/// One frame the way `fastgui-app` does it: layout at `logical` size, rasterize at `scale`,
/// then copy what changed into the texture.
fn frame(tree: &mut WidgetTree, target: &mut Target, logical: (f32, f32), scale: f32) -> Sample {
    let start = Instant::now();
    tree.compute_layout(logical.0, logical.1);
    let laid_out = Instant::now();
    let (pw, ph) = ((logical.0 * scale) as u32, (logical.1 * scale) as u32);
    let frame = black_box(target.chrome.rasterize(tree, pw, ph, None, scale));
    let rastered = Instant::now();
    let bytes = copy_to_texture(frame, &mut target.texture);
    Sample {
        layout: (laid_out - start).as_secs_f64() * 1e3,
        raster: (rastered - laid_out).as_secs_f64() * 1e3,
        copy: rastered.elapsed().as_secs_f64() * 1e3,
        bytes,
    }
}

const BUDGET_60HZ_MS: f64 = 1000.0 / 60.0;
const BUDGET_120HZ_MS: f64 = 1000.0 / 120.0;

fn print_header() {
    println!(
        "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>11}  {:>6}  verdict",
        "", "layout ms", "raster ms", "copy ms", "total ms", "upload KiB", "%60Hz"
    );
}

/// `one_off` rows (first frame) aren't judged against the per-frame budget.
fn print_sample(label: &str, s: &Sample, one_off: bool, worst: &mut Vec<(String, f64)>) {
    let total = s.layout + s.raster + s.copy;
    let share = total / BUDGET_60HZ_MS * 100.0;
    let verdict = if one_off {
        "one-off hitch"
    } else if total > BUDGET_60HZ_MS * 0.5 {
        "BOTTLENECK (>50% of a 60 Hz frame)"
    } else if total > BUDGET_120HZ_MS * 0.5 {
        "tight at 120 Hz"
    } else {
        "ok"
    };
    println!(
        "  {label:<14} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>11.1}  {:>5.0}%  {verdict}",
        s.layout,
        s.raster,
        s.copy,
        total,
        s.bytes as f64 / 1024.0,
        share
    );
    if !one_off {
        worst.push((label.to_string(), total));
    }
}

/// Averages `frames` frames after one warm-up frame.
fn time(label: &str, frames: usize, worst: &mut Vec<(String, f64)>, mut run: impl FnMut(usize) -> Sample) {
    run(0);
    let mut total = Sample::default();
    for i in 1..=frames {
        let s = run(i);
        total.layout += s.layout;
        total.raster += s.raster;
        total.copy += s.copy;
        total.bytes += s.bytes;
    }
    let n = frames as f64;
    let avg = Sample {
        layout: total.layout / n,
        raster: total.raster / n,
        copy: total.copy / n,
        bytes: (total.bytes as f64 / n) as u64,
    };
    print_sample(label, &avg, false, worst);
}

fn main() {
    println!(
        "fastgui chrome_bench — {} {}, {} threads available (chrome renders on one), {} build",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        if cfg!(debug_assertions) { "DEBUG (numbers meaningless — use --release)" } else { "release" },
    );
    println!("verdicts: ok < 4.2 ms < tight at 120 Hz < 8.3 ms < BOTTLENECK (half a 60 Hz frame)\n");
    let mut worst = Vec::new();

    for scale in [1.0f32, 2.0] {
        let size = (1280.0f32, 800.0f32);
        println!("dock scene, scale {scale}× ({}×{} px)", (size.0 * scale) as u32, (size.1 * scale) as u32);
        print_header();
        let Demo { mut tree, slider, readout } = build();
        let mut target = Target::new();
        print_sample("first frame", &frame(&mut tree, &mut target, size, scale), true, &mut worst);
        time("slider drag", 200, &mut worst, |i| {
            let value = (i % 100) as f32 / 100.0;
            tree.mutate_kind(slider, |kind| {
                if let WidgetKind::Slider { value: v, .. } = kind {
                    *v = value;
                }
            });
            set_label(&mut tree, readout, format!("Step: {value:.2}"));
            frame(&mut tree, &mut target, size, scale)
        });
        time("resize", 50, &mut worst, |i| frame(&mut tree, &mut target, (size.0 - (i % 2) as f32, size.1), scale));
        tag_scale(&mut worst, "dock", scale);
    }

    for scale in [1.0f32, 2.0] {
        let size = (1600.0f32, 1000.0f32);
        println!(
            "\ntable scene ({} cells), scale {scale}× ({}×{} px)",
            TABLE_ROWS * TABLE_COLS,
            (size.0 * scale) as u32,
            (size.1 * scale) as u32
        );
        print_header();
        let Table { mut tree, spacer, cells } = build_table();
        let mut target = Target::new();
        print_sample("first frame", &frame(&mut tree, &mut target, size, scale), true, &mut worst);
        let mut tick = 0u64;
        time("1 cell ticks", 100, &mut worst, |_| {
            tick += 1;
            set_label(&mut tree, cells[TABLE_COLS + 3], format!("{:.2}", tick as f64 * 0.37));
            frame(&mut tree, &mut target, size, scale)
        });
        time("50 cells tick", 100, &mut worst, |_| {
            tick += 1;
            for (k, &cell) in cells.iter().step_by(TABLE_COLS).enumerate() {
                set_label(&mut tree, cell, format!("{:.2}", (tick * 50 + k as u64) as f64 * 0.37));
            }
            frame(&mut tree, &mut target, size, scale)
        });
        time("scroll", 50, &mut worst, |i| {
            tree.set_style(
                spacer,
                Style {
                    size: Size { width: Dimension::auto(), height: length((i % 20) as f32) },
                    flex_shrink: 0.0,
                    ..Default::default()
                },
            );
            frame(&mut tree, &mut target, size, scale)
        });
        time("resize warm", 50, &mut worst, |i| frame(&mut tree, &mut target, (size.0 - (i % 2) as f32, size.1), scale));
        time("resize drag", 50, &mut worst, |i| {
            frame(&mut tree, &mut target, (size.0 - 2.0 - i as f32, size.1), scale)
        });
        tag_scale(&mut worst, "table", scale);
    }

    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\nslowest per-frame cases:");
    for (label, total) in worst.iter().take(3) {
        println!("  {label:<26} {total:7.3} ms  ({:.0}% of a 60 Hz frame)", total / BUDGET_60HZ_MS * 100.0);
    }
    let over: Vec<_> = worst.iter().filter(|(_, t)| *t > BUDGET_60HZ_MS * 0.5).collect();
    if over.is_empty() {
        println!("\nresult: CPU chrome is not a bottleneck on this machine (every case < half a 60 Hz frame).");
    } else {
        println!(
            "\nresult: CPU chrome IS a bottleneck here for: {}",
            over.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join(", ")
        );
    }
}

/// Prefix the rows `time` just recorded with their scene and scale, for the summary.
fn tag_scale(worst: &mut [(String, f64)], scene: &str, scale: f32) {
    for (label, _) in worst.iter_mut().filter(|(l, _)| !l.contains('×')) {
        *label = format!("{scene} {scale}× {label}");
    }
}
