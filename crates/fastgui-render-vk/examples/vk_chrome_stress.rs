//! Runs the Vulkan backend's real window + swapchain path with a 2,000-cell table whose content
//! and layout change every frame (cell values tick, a slider moves, the table scrolls), so the
//! chrome's quads are re-sent and atlas slots churn continuously. Watch stderr: the renderer
//! prints any validation-layer message it receives. Close the window to exit.
//!
//!   cargo run --release -p fastgui-render-vk --example vk_chrome_stress
//!
//! macOS (MoltenVK via Homebrew), with validation:
//!
//!   DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
//!   VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d \
//!   cargo run -p fastgui-render-vk --example vk_chrome_stress

use std::time::Duration;

use fastgui_core::taffy::prelude::*;
use fastgui_core::widget::{Color, WidgetKind, WidgetTree};
use fastgui_core::{command_channel, Readback};
use fastgui_render_vk::{Command, CommandDispatch, EventWaker, RenderThreadHandles};

const ROWS: usize = 50;
const COLS: usize = 40;

fn main() {
    let (sender, commands) = command_channel();
    let waker = EventWaker::default();
    let clear_color = [0.1, 0.11, 0.13, 1.0];
    let handles = RenderThreadHandles { commands, clear_color: Readback::new(clear_color), waker: waker.clone() };
    let dispatch = CommandDispatch { sender, waker, floating_region: None };

    // Ids come back from the render thread's tree through this channel once it's built.
    let (ids_tx, ids_rx) = std::sync::mpsc::channel();
    dispatch
        .send(Command::MutateWidgetTree(Box::new(move |tree: &mut WidgetTree| {
            let text = Color([0.9, 0.9, 0.92, 1.0]);
            let root = tree.root();
            let spacer = tree.new_node(
                Style { size: Size { width: Dimension::auto(), height: length(0.0_f32) }, flex_shrink: 0.0, ..Default::default() },
                WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
            );
            tree.add_child(root, spacer);
            let slider = tree.new_node(
                Style { size: Size { width: Dimension::auto(), height: length(24.0_f32) }, flex_shrink: 0.0, ..Default::default() },
                WidgetKind::Slider {
                    value: 0.0,
                    min: 0.0,
                    max: 1.0,
                    track_color: Color([0.3, 0.3, 0.35, 1.0]),
                    thumb_color: Color([0.4, 0.65, 1.0, 1.0]),
                    on_change: None,
                },
            );
            tree.add_child(root, slider);
            let mut cells = Vec::new();
            for r in 0..ROWS {
                let background = if r % 2 == 0 { Color([0.14, 0.15, 0.17, 1.0]) } else { Color([0.12, 0.13, 0.15, 1.0]) };
                let row = tree.new_node(
                    Style { flex_direction: FlexDirection::Row, size: Size { width: Dimension::auto(), height: length(18.0_f32) }, flex_shrink: 0.0, ..Default::default() },
                    WidgetKind::Container { background, region_id: None },
                );
                tree.add_child(root, row);
                for c in 0..COLS {
                    let cell = tree.new_node(
                        Style { flex_grow: 1.0, flex_basis: length(0.0_f32), min_size: Size { width: length(0.0_f32), height: Dimension::auto() }, ..Default::default() },
                        WidgetKind::Label { text: format!("{}.{c:02}", r * 7 + c), font_size: 11.0, color: text },
                    );
                    tree.add_child(row, cell);
                    cells.push(cell);
                }
            }
            let _ = ids_tx.send((spacer, slider, cells));
        })))
        .expect("render thread alive");

    std::thread::spawn(move || {
        let (spacer, slider, cells) = ids_rx.recv().expect("tree built once run() starts");
        for tick in 0u64.. {
            std::thread::sleep(Duration::from_millis(8));
            let changed: Vec<_> = cells.iter().copied().skip((tick % 40) as usize).step_by(40).collect();
            let sent = dispatch.send(Command::MutateWidgetTree(Box::new(move |tree: &mut WidgetTree| {
                tree.set_style(
                    spacer,
                    Style { size: Size { width: Dimension::auto(), height: length((tick % 30) as f32) }, flex_shrink: 0.0, ..Default::default() },
                );
                tree.mutate_kind(slider, |k| {
                    if let WidgetKind::Slider { value, .. } = k {
                        *value = (tick % 200) as f32 / 200.0;
                    }
                });
                for (k, cell) in changed.into_iter().enumerate() {
                    tree.mutate_kind(cell, |kind| {
                        if let WidgetKind::Label { text, .. } = kind {
                            *text = format!("{:.2}", (tick * 50 + k as u64) as f64 * 0.37);
                        }
                    });
                }
            })));
            if sent.is_err() {
                break; // window closed
            }
        }
    });

    fastgui_render_vk::run("fastgui Vulkan chrome stress", 1280, 900, clear_color, handles).expect("run");
}
