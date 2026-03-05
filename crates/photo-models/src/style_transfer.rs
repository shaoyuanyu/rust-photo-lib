use std::path::{Path, PathBuf};

use photo_core::{ImageFrame, PhotoError, Result};
use photo_imageops::{image_to_nchw_f32, nchw_f32_to_image, resize_rgb};
use photo_onnx::{NamedTensor, OnnxEngine, SessionOptions};

#[derive(Debug, Clone, Copy)]
pub enum StyleTransferNormalization {
    ZeroOne,
    MinusOneOne,
}

impl StyleTransferNormalization {
    fn is_minus1_1(self) -> bool {
        matches!(self, Self::MinusOneOne)
    }
}

#[derive(Debug, Clone)]
pub struct StyleTransferModel {
    pub model_path: PathBuf,
    pub input_name: String,
    pub output_name: Option<String>,
    pub input_size: Option<(u32, u32)>,
    pub normalization: StyleTransferNormalization,
}

impl StyleTransferModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            input_name: "input".to_string(),
            output_name: None,
            input_size: None,
            normalization: StyleTransferNormalization::MinusOneOne,
        }
    }

    pub fn run<E: OnnxEngine>(&self, engine: &mut E, image: &ImageFrame) -> Result<ImageFrame> {
        let (target_w, target_h) = self.input_size.unwrap_or((image.width, image.height));
        let resized = if target_w != image.width || target_h != image.height {
            resize_rgb(image, target_w, target_h)?
        } else {
            image.clone()
        };

        engine
            .load_model(&self.model_path, SessionOptions::default())
            .map_err(|e| PhotoError::Model(e.to_string()))?;

        let tensor_data = image_to_nchw_f32(&resized, self.normalization.is_minus1_1());
        let input = NamedTensor {
            name: self.input_name.clone(),
            shape: vec![1, 3, target_h as usize, target_w as usize],
            data: tensor_data,
        };

        let outputs = engine
            .run(&[input])
            .map_err(|e| PhotoError::Model(e.to_string()))?;

        let output = if let Some(name) = &self.output_name {
            outputs
                .iter()
                .find(|t| &t.name == name)
                .cloned()
                .ok_or_else(|| PhotoError::Model(format!("output tensor `{name}` not found")))?
        } else {
            outputs
                .first()
                .cloned()
                .ok_or_else(|| PhotoError::Model("model returned no outputs".into()))?
        };

        if output.shape.len() != 4 || output.shape[0] != 1 || output.shape[1] != 3 {
            return Err(PhotoError::Model(format!(
                "unexpected output shape: {:?}, expected [1,3,H,W]",
                output.shape
            )));
        }

        let out_h = output.shape[2] as u32;
        let out_w = output.shape[3] as u32;
        let out_img =
            nchw_f32_to_image(out_w, out_h, &output.data, self.normalization.is_minus1_1())?;

        if out_w != image.width || out_h != image.height {
            resize_rgb(&out_img, image.width, image.height)
        } else {
            Ok(out_img)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use photo_onnx::{OnnxError, Result as OnnxResult};

    use super::*;

    struct MockEngine {
        output_shape: Vec<usize>,
        output_data: Vec<f32>,
    }

    impl OnnxEngine for MockEngine {
        fn backend_name(&self) -> &'static str {
            "mock"
        }

        fn load_model<P: AsRef<Path>>(
            &mut self,
            _path: P,
            _options: SessionOptions,
        ) -> OnnxResult<()> {
            Ok(())
        }

        fn run(&mut self, _inputs: &[NamedTensor]) -> OnnxResult<Vec<NamedTensor>> {
            Ok(vec![NamedTensor {
                name: "output_0".to_string(),
                shape: self.output_shape.clone(),
                data: self.output_data.clone(),
            }])
        }
    }

    #[test]
    fn style_transfer_returns_same_size_image() {
        let input = ImageFrame::new(
            2,
            2,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        )
        .expect("valid frame");

        let mut engine = MockEngine {
            output_shape: vec![1, 3, 2, 2],
            output_data: vec![
                0.0, 0.2, 0.4, 0.6, // R
                0.1, 0.3, 0.5, 0.7, // G
                0.2, 0.4, 0.6, 0.8, // B
            ],
        };

        let mut model = StyleTransferModel::new("dummy.onnx");
        model.normalization = StyleTransferNormalization::ZeroOne;

        let output = model.run(&mut engine, &input).expect("run ok");
        assert_eq!(output.width, 2);
        assert_eq!(output.height, 2);
        assert_eq!(output.data.len(), input.data.len());
    }

    #[test]
    fn style_transfer_rejects_invalid_shape() {
        let input = ImageFrame::new(1, 1, vec![10, 20, 30]).expect("valid frame");
        let mut engine = MockEngine {
            output_shape: vec![1, 1, 1, 1],
            output_data: vec![0.0],
        };

        let model = StyleTransferModel::new("dummy.onnx");
        let err = model
            .run(&mut engine, &input)
            .expect_err("should fail on shape");
        match err {
            PhotoError::Model(_) => {}
            _ => panic!("expected model error"),
        }
    }
}
