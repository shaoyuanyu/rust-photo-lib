//! 照片处理的核心数据结构与管线抽象。

use std::fmt::Debug;

use thiserror::Error;

/// photo-core 的便捷结果类型。
pub type Result<T> = std::result::Result<T, PhotoError>;

/// 核心组件返回的错误类型。
#[derive(Debug, Error)]
pub enum PhotoError {
    /// 输入图像数据不合法或不一致。
    #[error("invalid image data: {0}")]
    InvalidImageData(String),
    /// 处理管线阶段失败。
    #[error("pipeline error: {0}")]
    Pipeline(String),
    /// 模型相关错误。
    #[error("model error: {0}")]
    Model(String),
}

/// 内存中的 RGB 图像帧。
#[derive(Clone, Debug)]
pub struct ImageFrame {
    /// 像素宽度。
    pub width: u32,
    /// 像素高度。
    pub height: u32,
    /// RGB 字节缓冲区（width * height * 3）。
    pub data: Vec<u8>,
}

impl ImageFrame {
    /// 创建新的 RGB 帧，并校验缓冲区长度。
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(PhotoError::InvalidImageData(format!(
                "expected {} bytes (RGB), got {}",
                expected,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BasicAdjustments {
    /// 曝光（EV）。
    pub exposure: f32,
    /// 线性亮度偏移。
    pub brightness: f32,
    /// 以中灰为中心的对比度缩放。
    pub contrast: f32,
    /// 饱和度倍数。
    pub saturation: f32,
    /// 白平衡色温偏移。
    pub temperature: f32,
    /// 绿-洋红色调偏移。
    pub tint: f32,
    /// 色相偏移（度）。
    pub hue_shift_degrees: f32,
}

impl Default for BasicAdjustments {
    /// 默认的中性调整值。
    fn default() -> Self {
        Self {
            exposure: 0.0,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
            tint: 0.0,
            hue_shift_degrees: 0.0,
        }
    }
}

/// 处理管线中的单个阶段。
pub trait Stage: Debug + Send + Sync {
    /// 人类可读的阶段名称。
    fn name(&self) -> &'static str;
    /// 对输入帧应用处理。
    fn apply(&self, input: ImageFrame) -> Result<ImageFrame>;
}

/// 按顺序执行的阶段集合。
#[derive(Default)]
pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
}

impl Pipeline {
    /// 创建空的处理管线。
    pub fn new() -> Self {
        Self::default()
    }

    /// 向管线追加一个阶段。
    pub fn push<S>(&mut self, stage: S)
    where
        S: Stage + 'static,
    {
        self.stages.push(Box::new(stage));
    }

    /// 按顺序执行所有阶段。
    pub fn run(&self, mut frame: ImageFrame) -> Result<ImageFrame> {
        for stage in &self.stages {
            frame = stage.apply(frame)?;
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用阶段：对每个通道加 1。
    #[derive(Debug)]
    struct AddOneStage;

    impl Stage for AddOneStage {
        /// 测试阶段名称。
        fn name(&self) -> &'static str {
            "add-one"
        }

        /// 对每个字节执行饱和加 1。
        fn apply(&self, mut input: ImageFrame) -> Result<ImageFrame> {
            for value in &mut input.data {
                *value = value.saturating_add(1);
            }
            Ok(input)
        }
    }

    /// 验证非法 RGB 缓冲长度会被拒绝。
    #[test]
    fn image_frame_validates_rgb_len() {
        let result = ImageFrame::new(2, 2, vec![0; 3]);
        assert!(result.is_err());
    }

    /// 验证管线按顺序执行阶段。
    #[test]
    fn pipeline_runs_all_stages() {
        let mut pipeline = Pipeline::new();
        pipeline.push(AddOneStage);
        pipeline.push(AddOneStage);

        let input = ImageFrame::new(1, 1, vec![1, 2, 3]).expect("valid frame");
        let output = pipeline.run(input).expect("pipeline run");

        assert_eq!(output.data, vec![3, 4, 5]);
    }
}
