//! `Viewport.submit_d3d11` / `Image.submit_d3d11`. Windows duplicates the NT handles into the
//! import mailbox, so the caller keeps and closes their own. Other platforms refuse before
//! anything is queued.

use std::sync::Arc;

use fastgui_core::{ImportStatus, ImportedLayer};
#[cfg(windows)]
use fastgui_core::{D3d11Fence, D3d11Import, SharedFormat};
#[cfg(not(windows))]
use pyo3::exceptions::PyRuntimeError;
#[cfg(windows)]
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// What `submit_d3d11` was given, beyond the texture handle and size.
pub(crate) struct Options<'a> {
    pub format: Option<&'a str>,
    pub fence: Option<isize>,
    pub wait_value: u64,
    pub signal_value: u64,
}

pub(crate) fn prepare(
    handle: isize,
    width: u32,
    height: u32,
    options: Options<'_>,
    status: &Arc<ImportStatus>,
) -> PyResult<ImportedLayer> {
    #[cfg(not(windows))]
    {
        let _ = (handle, width, height, options, status);
        Err(PyRuntimeError::new_err(
            "submit_d3d11 is only available on Windows. Use submit_frame, submit_cuda, submit_metal, or submit_dmabuf.",
        ))
    }
    #[cfg(windows)]
    {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("D3D11 texture must be at least 1x1"));
        }
        let format = match options.format.map(|f| f.trim().to_ascii_lowercase()) {
            None => None,
            Some(f) if f == "rgba8" => Some(SharedFormat::Rgba8),
            Some(f) if f == "bgra8" => Some(SharedFormat::Bgra8),
            Some(_) => return Err(PyValueError::new_err("format must be None, 'rgba8', or 'bgra8'")),
        };
        if options.fence.is_none() && (options.wait_value != 0 || options.signal_value != 0) {
            return Err(PyValueError::new_err("wait_value and signal_value need a fence"));
        }
        let duped = dup_handle(handle).map_err(|_| {
            PyValueError::new_err("handle is not a valid NT handle (use IDXGIResource1::CreateSharedHandle)")
        })?;
        // Owned from here, so an error below closes it.
        let texture = D3d11Import::new(duped, width, height)
            .with_format(format)
            .with_status(status.clone());
        let fence = match options.fence {
            None => None,
            Some(fence) => {
                let duped = dup_handle(fence).map_err(|_| {
                    PyValueError::new_err("fence is not a valid NT handle (use ID3D11Fence::CreateSharedHandle)")
                })?;
                Some(D3d11Fence::new(duped, options.wait_value, options.signal_value))
            }
        };
        Ok(ImportedLayer::D3d11(texture.with_fence(fence)))
    }
}

/// The adapter LUID (`low | high << 32`) of the GPU a window renders on, so a D3D device can be
/// created on it. `None` off Windows or when it can't be determined.
pub(crate) fn adapter_luid() -> Option<u64> {
    crate::backend::adapter_luid()
}

#[cfg(windows)]
fn dup_handle(handle: isize) -> Result<isize, ()> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn DuplicateHandle(
            source_process: isize,
            source: isize,
            target_process: isize,
            target: *mut isize,
            access: u32,
            inherit: i32,
            options: u32,
        ) -> i32;
    }
    const DUPLICATE_SAME_ACCESS: u32 = 2;
    // 0 is not a handle, and -1 is the current-process pseudo-handle, which would "duplicate" fine.
    if handle == 0 || handle == -1 {
        return Err(());
    }
    let mut out = 0isize;
    let process = unsafe { GetCurrentProcess() };
    let ok = unsafe { DuplicateHandle(process, handle, process, &mut out, 0, 0, DUPLICATE_SAME_ACCESS) };
    if ok == 0 {
        Err(())
    } else {
        Ok(out)
    }
}
