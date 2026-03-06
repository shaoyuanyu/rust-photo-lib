use std::fmt::Debug;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, PhotoError>;

#[derive(Debug, Error)]
pub enum PhotoError {
    #[error("invalid image data: {0}")]
    InvalidImageData(String),
    #[error("pipeline error: {0}")]
    Pipeline(String),
    #[error("model error: {0}")]
    Model(String),
}

#[derive(Clone, Debug)]
pub struct ImageFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl ImageFrame {
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
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub hue_shift_degrees: f32,
}

impl Default for BasicAdjustments {
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

pub trait Stage: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn apply(&self, input: ImageFrame) -> Result<ImageFrame>;
}

#[derive(Default)]
pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push<S>(&mut self, stage: S)
    where
        S: Stage + 'static,
    {
        self.stages.push(Box::new(stage));
    }

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

    #[derive(Debug)]
    struct AddOneStage;

    impl Stage for AddOneStage {
        fn name(&self) -> &'static str {
            "add-one"
        }

        fn apply(&self, mut input: ImageFrame) -> Result<ImageFrame> {
            for value in &mut input.data {
                *value = value.saturating_add(1);
            }
            Ok(input)
        }
    }

    #[test]
    fn image_frame_validates_rgb_len() {
        let result = ImageFrame::new(2, 2, vec![0; 3]);
        assert!(result.is_err());
    }

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
