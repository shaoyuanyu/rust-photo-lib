use std::path::Path;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, OnnxError>;

#[derive(Debug, Clone)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct SessionOptions {
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("model not loaded")]
    ModelNotLoaded,
    #[error("backend not available: {0}")]
    BackendUnavailable(String),
    #[error("invalid tensor: {0}")]
    InvalidTensor(String),
    #[error("inference failed: {0}")]
    Inference(String),
}

pub trait OnnxEngine {
    fn backend_name(&self) -> &'static str;
    fn load_model<P: AsRef<Path>>(&mut self, path: P, options: SessionOptions) -> Result<()>;
    fn run(&mut self, inputs: &[NamedTensor]) -> Result<Vec<NamedTensor>>;
}
