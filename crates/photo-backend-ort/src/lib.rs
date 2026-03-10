//! ONNX Runtime 后端实现。

use std::path::{Path, PathBuf};

use photo_onnx::{NamedTensor, OnnxEngine, OnnxError, Result, SessionOptions};

/// ONNX Runtime 引擎封装。
#[derive(Default)]
pub struct OrtEngine {
    /// 已加载的模型路径。
    model_path: Option<PathBuf>,
    #[cfg(feature = "backend")]
    /// 启用后端特性时的 ORT 会话。
    session: Option<ort::session::Session>,
}

impl OnnxEngine for OrtEngine {
    /// 后端标识。
    fn backend_name(&self) -> &'static str {
        "ort"
    }

    /// 加载 ONNX 模型到 ORT。
    fn load_model<P: AsRef<Path>>(&mut self, path: P, _options: SessionOptions) -> Result<()> {
        let model_path = path.as_ref().to_path_buf();

        #[cfg(feature = "backend")]
        {
            let session = ort::session::Session::builder()
                .map_err(|err| OnnxError::Inference(err.to_string()))?
                .commit_from_file(&model_path)
                .map_err(|err| OnnxError::Inference(err.to_string()))?;
            self.session = Some(session);
            self.model_path = Some(model_path);
            return Ok(());
        }

        #[cfg(not(feature = "backend"))]
        {
            let _ = &model_path;
            Err(OnnxError::BackendUnavailable(
                "photo-backend-ort built without `backend` feature".into(),
            ))
        }
    }

    /// 使用已加载模型执行推理。
    fn run(&mut self, inputs: &[NamedTensor]) -> Result<Vec<NamedTensor>> {
        if self.model_path.is_none() {
            return Err(OnnxError::ModelNotLoaded);
        }

        #[cfg(feature = "backend")]
        {
            let session = self.session.as_mut().ok_or(OnnxError::ModelNotLoaded)?;
            if inputs.is_empty() {
                return Err(OnnxError::InvalidTensor("inputs must not be empty".into()));
            }

            let mut ort_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let tensor =
                    ort::value::Tensor::from_array((input.shape.clone(), input.data.clone()))
                        .map_err(|err| OnnxError::InvalidTensor(err.to_string()))?;
                ort_inputs.push((input.name.clone(), tensor));
            }

            let outputs = session
                .run(ort_inputs)
                .map_err(|err| OnnxError::Inference(err.to_string()))?;

            let mut named = Vec::with_capacity(outputs.len());
            for (name, out) in outputs.into_iter() {
                let tensor = out
                    .try_extract_array::<f32>()
                    .map_err(|err| OnnxError::Inference(err.to_string()))?;
                named.push(NamedTensor {
                    name,
                    shape: tensor.shape().to_vec(),
                    data: tensor.iter().copied().collect(),
                });
            }
            return Ok(named);
        }

        #[cfg(not(feature = "backend"))]
        {
            let _ = inputs;
            Err(OnnxError::BackendUnavailable(
                "photo-backend-ort built without `backend` feature".into(),
            ))
        }
    }
}
