//! `Viewport.submit_dmabuf` / `Image.submit_dmabuf`. Linux dups the fd into the import mailbox.
//! Other platforms refuse before anything is queued.

#[cfg(target_os = "linux")]
use fastgui_core::DmaBufImport;
use fastgui_core::ImportedLayer;
#[cfg(not(target_os = "linux"))]
use pyo3::exceptions::PyRuntimeError;
#[cfg(target_os = "linux")]
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub(crate) fn prepare(fd: i32, width: u32, height: u32, stride: u32, modifier: u64) -> PyResult<ImportedLayer> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (fd, width, height, stride, modifier);
        Err(PyRuntimeError::new_err(
            "submit_dmabuf is only available on Linux. Use submit_frame, submit_cuda, or submit_metal.",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("dma-buf image must be at least 1x1"));
        }
        if modifier != 0 {
            return Err(PyValueError::new_err(
                "DRM format modifiers are not imported yet; pass modifier 0 for a linear RGBA8 dma-buf",
            ));
        }
        if stride != 0 && stride < width.saturating_mul(4) {
            return Err(PyValueError::new_err("stride must be 0 (tightly packed) or at least width * 4"));
        }
        let duped = dup_fd(fd).map_err(|_| PyValueError::new_err("failed to duplicate the dma-buf fd"))?;
        Ok(ImportedLayer::DmaBuf(DmaBufImport::new(duped, width, height, stride, modifier)))
    }
}

#[cfg(target_os = "linux")]
fn dup_fd(fd: i32) -> Result<i32, ()> {
    extern "C" {
        fn dup(fd: i32) -> i32;
    }
    let owned = unsafe { dup(fd) };
    if owned < 0 {
        Err(())
    } else {
        Ok(owned)
    }
}
