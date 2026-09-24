use winit::window::Window;

/// Hide a shown window from the compositor (Windows DWM cloaking) without making it invisible
/// to the system — the backend can still present to it. Used to keep a new window off screen
/// until its first frame lands: a plain hidden-then-shown window is composited the moment it's
/// shown, and Windows paints it white until that first present. No-op elsewhere.
#[cfg(target_os = "windows")]
pub fn set_cloaked(window: &Window, cloaked: bool) {
    use windows_sys::Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_CLOAK};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let Ok(handle) = window.window_handle() else { return };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else { return };
    let value: i32 = cloaked.into();
    // Best effort: if cloaking fails the window just shows as it did before.
    unsafe {
        DwmSetWindowAttribute(
            win32.hwnd.get(),
            DWMWA_CLOAK as u32,
            std::ptr::from_ref(&value).cast(),
            std::mem::size_of::<i32>() as u32,
        );
    }
}

#[cfg(not(target_os = "windows"))]
pub fn set_cloaked(_window: &Window, _cloaked: bool) {}
