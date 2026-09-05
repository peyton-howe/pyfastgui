use crate::sys::CUresult;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("failed to load nvcuda.dll: {0}")]
    Load(#[from] libloading::Error),
    #[error("nvcuda.dll is missing the expected entry point {0}")]
    MissingSymbol(&'static str, #[source] libloading::Error),
    #[error("CUDA driver call failed: {message} (CUresult {code})")]
    Driver { code: CUresult, message: String },
}

impl CudaError {
    pub(crate) fn missing(name: &'static str, err: libloading::Error) -> Self {
        Self::MissingSymbol(name, err)
    }
}
