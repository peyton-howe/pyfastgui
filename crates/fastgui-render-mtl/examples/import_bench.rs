fn main() {
    #[cfg(target_os = "macos")]
    fastgui_render_mtl::run_import_window_bench();
}
