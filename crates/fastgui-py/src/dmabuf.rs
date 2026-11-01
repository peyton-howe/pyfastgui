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
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    const F_DUPFD_CLOEXEC: i32 = 1030;
    // SAFETY: `fd` must refer to the caller's open descriptor; fcntl rejects invalid fds.
    // The fresh duplicate is uniquely owned and passed to DmaBufImport, which closes it on drop
    // unless ownership is transferred to the driver.
    let owned = unsafe { fcntl(fd, F_DUPFD_CLOEXEC, 0) };
    if owned < 0 {
        Err(())
    } else {
        Ok(owned)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::{ErrorKind, Read};
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;

    #[test]
    fn duplicate_is_close_on_exec_and_owned_by_import() {
        extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        const F_GETFD: i32 = 1;
        const FD_CLOEXEC: i32 = 1;

        let (source, mut peer) = UnixStream::pair().unwrap();
        peer.set_nonblocking(true).unwrap();
        let import = DmaBufImport::new(dup_fd(source.as_raw_fd()).unwrap(), 1, 1, 4, 0);
        // SAFETY: `import` keeps the duplicated fd open for this query.
        let flags = unsafe { fcntl(import.fd(), F_GETFD) };
        assert!(flags >= 0);
        assert_ne!(flags & FD_CLOEXEC, 0);
        drop(source);
        let mut byte = [0];
        assert_eq!(peer.read(&mut byte).unwrap_err().kind(), ErrorKind::WouldBlock);
        drop(import);
        assert_eq!(peer.read(&mut byte).unwrap(), 0);
        assert!(dup_fd(-1).is_err());
    }
}
