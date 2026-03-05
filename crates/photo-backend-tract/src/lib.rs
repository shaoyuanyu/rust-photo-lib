use std::path::{Path, PathBuf};

use photo_onnx::{NamedTensor, OnnxEngine, OnnxError, Result, SessionOptions};

#[derive(Default)]
pub struct TractEngine {
    model_path: Option<PathBuf>,
    #[cfg(feature = "backend")]
    runnable: Option<
        tract_onnx::prelude::SimplePlan<
            tract_onnx::prelude::TypedFact,
            Box<dyn tract_onnx::prelude::TypedOp>,
            tract_onnx::prelude::Graph<
                tract_onnx::prelude::TypedFact,
                Box<dyn tract_onnx::prelude::TypedOp>,
            >,
        >,
    >,
}

impl OnnxEngine for TractEngine {
    fn backend_name(&self) -> &'static str {
        "tract"
    }

    fn load_model<P: AsRef<Path>>(&mut self, path: P, _options: SessionOptions) -> Result<()> {
        let model_path = path.as_ref().to_path_buf();

        #[cfg(feature = "backend")]
        {
            use tract_onnx::prelude::*;
            let runnable = tract_onnx::onnx()
                .model_for_path(&model_path)
                .map_err(|err| OnnxError::Inference(err.to_string()))?
                .into_optimized()
                .map_err(|err| OnnxError::Inference(err.to_string()))?
                .into_runnable()
                .map_err(|err| OnnxError::Inference(err.to_string()))?;
            self.runnable = Some(runnable);
            self.model_path = Some(model_path);
            return Ok(());
        }

        #[cfg(not(feature = "backend"))]
        {
            let _ = &model_path;
            Err(OnnxError::BackendUnavailable(
                "photo-backend-tract built without `backend` feature".into(),
            ))
        }
    }

    fn run(&mut self, inputs: &[NamedTensor]) -> Result<Vec<NamedTensor>> {
        if self.model_path.is_none() {
            return Err(OnnxError::ModelNotLoaded);
        }

        #[cfg(feature = "backend")]
        {
            use tract_onnx::prelude::*;

            let runnable = self.runnable.as_ref().ok_or(OnnxError::ModelNotLoaded)?;
            if inputs.is_empty() {
                return Err(OnnxError::InvalidTensor("inputs must not be empty".into()));
            }

            let mut tv = tvec![];
            for input in inputs {
                let array =
                    ndarray::ArrayD::from_shape_vec(input.shape.clone(), input.data.clone())
                        .map_err(|err| OnnxError::InvalidTensor(err.to_string()))?;
                let tensor = array.into_tensor();
                tv.push(tensor.into());
            }

            let outputs = runnable
                .run(tv)
                .map_err(|err| OnnxError::Inference(err.to_string()))?;

            let mut named = Vec::with_capacity(outputs.len());
            for (idx, output) in outputs.into_iter().enumerate() {
                let view = output
                    .to_array_view::<f32>()
                    .map_err(|err| OnnxError::Inference(err.to_string()))?;
                named.push(NamedTensor {
                    name: format!("output_{idx}"),
                    shape: view.shape().to_vec(),
                    data: view.iter().copied().collect(),
                });
            }
            return Ok(named);
        }

        #[cfg(not(feature = "backend"))]
        {
            let _ = inputs;
            Err(OnnxError::BackendUnavailable(
                "photo-backend-tract built without `backend` feature".into(),
            ))
        }
    }
}
