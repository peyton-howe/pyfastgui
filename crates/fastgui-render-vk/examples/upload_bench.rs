//! Headless Vulkan frame-upload timings. See `upload_bench` in the library.
//!
//! ```text
//! cargo run -p fastgui-render-vk --release --example upload_bench
//! ```

fn main() {
    fastgui_render_vk::run_upload_bench();
}
