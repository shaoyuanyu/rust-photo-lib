//! ONNX 推理接口抽象。

use std::path::Path;

use thiserror::Error;

/// ONNX 操作的便捷结果类型。
pub type Result<T> = std::result::Result<T, OnnxError>;

/// 推理输入/输出的具名张量容器。
#[derive(Debug, Clone)]
pub struct NamedTensor {
    /// 张量名称（输入或输出）。
    pub name: String,
    /// 形状（NCHW 等顺序）。
    pub shape: Vec<usize>,
    /// 扁平数据缓冲区。
    pub data: Vec<f32>,
}

/// 后端会话配置。
#[derive(Debug, Clone, Default)]
pub struct SessionOptions {
    /// 算子内线程数。
    pub intra_threads: Option<usize>,
    /// 算子间线程数。
    pub inter_threads: Option<usize>,
}

/// ONNX 后端错误。
#[derive(Debug, Error)]
pub enum OnnxError {
    /// 模型尚未加载。
    #[error("model not loaded")]
    ModelNotLoaded,
    /// 后端未编译或不可用。
    #[error("backend not available: {0}")]
    BackendUnavailable(String),
    /// 输入或输出张量不合法。
    #[error("invalid tensor: {0}")]
    InvalidTensor(String),
    /// 推理执行失败。
    #[error("inference failed: {0}")]
    Inference(String),
}

/// ONNX 推理后端的抽象接口。
pub trait OnnxEngine {
    /// 后端标识。
    fn backend_name(&self) -> &'static str;
    /// 加载模型文件到后端。
    fn load_model<P: AsRef<Path>>(&mut self, path: P, options: SessionOptions) -> Result<()>;
    /// 使用给定输入运行推理。
    fn run(&mut self, inputs: &[NamedTensor]) -> Result<Vec<NamedTensor>>;
}
