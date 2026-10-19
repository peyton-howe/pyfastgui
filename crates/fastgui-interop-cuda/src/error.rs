use crate::sys::CUresult;

#[derive(Clone, Debug, thiserror::Error)]
pub enum CudaError {
    #[error("failed to load the CUDA driver ({library}): {message}")]
    Load { library: &'static str, message: String },
    #[error("the CUDA driver is missing the expected entry point {0}")]
    MissingSymbol(&'static str),
    #[error("CUDA driver call failed: {message} (CUresult {code})")]
    Driver { code: CUresult, message: String },
}
