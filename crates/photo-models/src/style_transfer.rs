//! 风格迁移模型封装与执行逻辑。

use std::path::{Path, PathBuf};

use photo_core::{ImageFrame, PhotoError, Result};
use photo_imageops::{
    TensorNormalization, fit_within_max_and_multiple, image_to_nchw_f32_with_normalization,
    nchw_f32_to_image_with_normalization, resize_rgb,
};
use photo_onnx::{NamedTensor, OnnxEngine, SessionOptions};

/// 风格迁移模型的归一化模式。
#[derive(Debug, Clone, Copy)]
pub enum StyleTransferNormalization {
    /// 0..1 归一化。
    ZeroOne,
    /// -1..1 归一化。
    MinusOneOne,
    /// 0..255 归一化。
    ZeroTwoFiftyFive,
}

impl StyleTransferNormalization {
    /// 映射到图像工具层的归一化类型。
    fn as_tensor_normalization(self) -> TensorNormalization {
        match self {
            Self::ZeroOne => TensorNormalization::ZeroOne,
            Self::MinusOneOne => TensorNormalization::MinusOneOne,
            Self::ZeroTwoFiftyFive => TensorNormalization::ZeroTwoFiftyFive,
        }
    }
}

/// 模型输入尺寸的缩放策略。
#[derive(Debug, Clone, Copy)]
pub enum StyleTransferResizePolicy {
    /// 保持原始尺寸。
    Original,
    /// 限制最大边并对齐到倍数。
    MaxDimensionMultiple { max_dim: u32, multiple: u32 },
}

impl StyleTransferResizePolicy {
    /// 计算目标尺寸。
    fn target_size(self, source_width: u32, source_height: u32) -> (u32, u32) {
        match self {
            Self::Original => (source_width, source_height),
            Self::MaxDimensionMultiple { max_dim, multiple } => {
                fit_within_max_and_multiple(source_width, source_height, max_dim, multiple)
            }
        }
    }
}

/// 风格迁移模型配置。
#[derive(Debug, Clone)]
pub struct StyleTransferModel {
    /// ONNX 模型路径。
    pub model_path: PathBuf,
    /// 期望的输入张量名称。
    pub input_name: String,
    /// 可选的输出张量名称。
    pub output_name: Option<String>,
    /// 可选的固定输入尺寸。
    pub input_size: Option<(u32, u32)>,
    /// 输入缩放策略。
    pub resize_policy: StyleTransferResizePolicy,
    /// 输入/输出归一化方式。
    pub normalization: StyleTransferNormalization,
}

impl StyleTransferModel {
    /// 创建默认配置的模型封装。
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            input_name: "input1".to_string(),
            output_name: None,
            input_size: None,
            resize_policy: StyleTransferResizePolicy::MaxDimensionMultiple {
                max_dim: 800,
                multiple: 8,
            },
            normalization: StyleTransferNormalization::ZeroTwoFiftyFive,
        }
    }

    /// 使用给定 ONNX 引擎执行风格迁移。
    pub fn run<E: OnnxEngine>(&self, engine: &mut E, image: &ImageFrame) -> Result<ImageFrame> {
        let (target_w, target_h) = if let Some(input_size) = self.input_size {
            input_size
        } else {
            self.resize_policy.target_size(image.width, image.height)
        };
        let resized = if target_w != image.width || target_h != image.height {
            resize_rgb(image, target_w, target_h)?
        } else {
            image.clone()
        };

        engine
            .load_model(&self.model_path, SessionOptions::default())
            .map_err(|e| PhotoError::Model(e.to_string()))?;

        let tensor_data = image_to_nchw_f32_with_normalization(
            &resized,
            self.normalization.as_tensor_normalization(),
        );
        let input_shape = vec![1, 3, target_h as usize, target_w as usize];

        let mut last_err = None;
        let mut outputs = None;
        for input_name in self.input_name_candidates() {
            let input = NamedTensor {
                name: input_name,
                shape: input_shape.clone(),
                data: tensor_data.clone(),
            };

            match engine.run(&[input]) {
                Ok(v) => {
                    outputs = Some(v);
                    break;
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }

        let outputs = outputs.ok_or_else(|| {
            let err_message = last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown inference error".to_string());
            PhotoError::Model(format!(
                "failed to run style transfer with candidate inputs {:?}: {err_message}",
                self.input_name_candidates()
            ))
        })?;

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
        let out_img = nchw_f32_to_image_with_normalization(
            out_w,
            out_h,
            &output.data,
            self.normalization.as_tensor_normalization(),
        )?;

        if out_w != image.width || out_h != image.height {
            resize_rgb(&out_img, image.width, image.height)
        } else {
            Ok(out_img)
        }
    }

    /// 构建输入名称候选列表以提升兼容性。
    fn input_name_candidates(&self) -> Vec<String> {
        const FALLBACKS: &[&str] = &["input1", "input", "image", "x"];

        let mut candidates = Vec::with_capacity(FALLBACKS.len() + 1);
        candidates.push(self.input_name.clone());
        for fallback in FALLBACKS {
            if !candidates.iter().any(|existing| existing == fallback) {
                candidates.push((*fallback).to_string());
            }
        }
        candidates
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use photo_onnx::{OnnxError, Result as OnnxResult};

    use super::*;

    /// 用于验证逻辑的模拟引擎。
    struct MockEngine {
        /// 可接受的输入名称（可选）。
        accepted_input_name: Option<String>,
        /// 模拟输出的形状。
        output_shape: Vec<usize>,
        /// 模拟输出的数据。
        output_data: Vec<f32>,
        /// 记录已见的输入名称。
        seen_input_names: Vec<String>,
        /// 记录已见的输入形状。
        seen_input_shape: Option<Vec<usize>>,
    }

    impl OnnxEngine for MockEngine {
        /// 后端标识。
        fn backend_name(&self) -> &'static str {
            "mock"
        }

        /// 模拟加载模型（无操作）。
        fn load_model<P: AsRef<Path>>(
            &mut self,
            _path: P,
            _options: SessionOptions,
        ) -> OnnxResult<()> {
            Ok(())
        }

        /// 校验输入并返回模拟输出。
        fn run(&mut self, inputs: &[NamedTensor]) -> OnnxResult<Vec<NamedTensor>> {
            let input = inputs
                .first()
                .ok_or_else(|| OnnxError::InvalidTensor("inputs must not be empty".into()))?;

            self.seen_input_names.push(input.name.clone());
            self.seen_input_shape = Some(input.shape.clone());

            if let Some(accepted_input_name) = &self.accepted_input_name {
                if &input.name != accepted_input_name {
                    return Err(OnnxError::Inference(format!(
                        "unexpected input name: {}",
                        input.name
                    )));
                }
            }

            Ok(vec![NamedTensor {
                name: "output_0".to_string(),
                shape: self.output_shape.clone(),
                data: self.output_data.clone(),
            }])
        }
    }

    /// 验证输出被调整回输入尺寸。
    #[test]
    fn style_transfer_returns_same_size_image() {
        let input = ImageFrame::new(
            2,
            2,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        )
        .expect("valid frame");

        let mut engine = MockEngine {
            accepted_input_name: None,
            output_shape: vec![1, 3, 2, 2],
            output_data: vec![
                0.0, 0.2, 0.4, 0.6, // R
                0.1, 0.3, 0.5, 0.7, // G
                0.2, 0.4, 0.6, 0.8, // B
            ],
            seen_input_names: vec![],
            seen_input_shape: None,
        };

        let mut model = StyleTransferModel::new("dummy.onnx");
        model.normalization = StyleTransferNormalization::ZeroOne;
        model.resize_policy = StyleTransferResizePolicy::Original;

        let output = model.run(&mut engine, &input).expect("run ok");
        assert_eq!(output.width, 2);
        assert_eq!(output.height, 2);
        assert_eq!(output.data.len(), input.data.len());
    }

    /// 验证非法输出形状会被拒绝。
    #[test]
    fn style_transfer_rejects_invalid_shape() {
        let input = ImageFrame::new(1, 1, vec![10, 20, 30]).expect("valid frame");
        let mut engine = MockEngine {
            accepted_input_name: None,
            output_shape: vec![1, 1, 1, 1],
            output_data: vec![0.0],
            seen_input_names: vec![],
            seen_input_shape: None,
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

    /// 验证默认配置的值。
    #[test]
    fn style_transfer_defaults_match_legacy_project_behavior() {
        let model = StyleTransferModel::new("dummy.onnx");
        assert_eq!(model.input_name, "input1");
        assert!(matches!(
            model.resize_policy,
            StyleTransferResizePolicy::MaxDimensionMultiple {
                max_dim: 800,
                multiple: 8
            }
        ));
        assert!(matches!(
            model.normalization,
            StyleTransferNormalization::ZeroTwoFiftyFive
        ));
    }

    /// 验证输入名称候选的重试顺序。
    #[test]
    fn style_transfer_retries_input_name_candidates() {
        let input = ImageFrame::new(1, 1, vec![10, 20, 30]).expect("valid frame");
        let mut engine = MockEngine {
            accepted_input_name: Some("input1".to_string()),
            output_shape: vec![1, 3, 1, 1],
            output_data: vec![10.0, 20.0, 30.0],
            seen_input_names: vec![],
            seen_input_shape: None,
        };

        let mut model = StyleTransferModel::new("dummy.onnx");
        model.input_name = "input".to_string();
        model.resize_policy = StyleTransferResizePolicy::Original;

        let output = model.run(&mut engine, &input).expect("run ok");
        assert_eq!(output.data, vec![10, 20, 30]);
        assert_eq!(engine.seen_input_names, vec!["input", "input1"]);
    }

    /// 验证默认缩放策略会限制最大边并对齐倍数。
    #[test]
    fn style_transfer_uses_default_max_dim_multiple_resize() {
        let input = ImageFrame::new(1600, 900, vec![128; 1600 * 900 * 3]).expect("valid frame");
        let mut engine = MockEngine {
            accepted_input_name: None,
            output_shape: vec![1, 3, 448, 800],
            output_data: vec![128.0; 448 * 800 * 3],
            seen_input_names: vec![],
            seen_input_shape: None,
        };

        let model = StyleTransferModel::new("dummy.onnx");
        let output = model.run(&mut engine, &input).expect("run ok");

        assert_eq!(output.width, 1600);
        assert_eq!(output.height, 900);
        assert_eq!(engine.seen_input_shape, Some(vec![1, 3, 448, 800]));
    }
}
