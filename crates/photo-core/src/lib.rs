use std::{collections::BTreeMap, fmt::Debug};

use serde::{Deserialize, Serialize};
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

pub const PHOTO_EDIT_RECIPE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GlobalAdjustments {
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub vibrance: f32,
    pub temperature: f32,
    pub tint: f32,
    #[serde(alias = "hue")]
    pub hue_shift_degrees: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToneCurve {
    pub points: Vec<[f32; 2]>,
}

impl ToneCurve {
    pub fn linear() -> Self {
        Self {
            points: vec![
                [0.0, 0.0],
                [0.25, 0.25],
                [0.5, 0.5],
                [0.75, 0.75],
                [1.0, 1.0],
            ],
        }
    }
}

impl Default for ToneCurve {
    fn default() -> Self {
        Self::linear()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HslColor {
    #[default]
    Red,
    Orange,
    Yellow,
    Green,
    Aqua,
    Blue,
    Purple,
    Magenta,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HslAdjustment {
    pub hue: f32,
    pub saturation: f32,
    pub luminance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalAdjustmentLayer {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_local_opacity")]
    pub opacity: f32,
    pub mask: MaskShape,
    #[serde(default)]
    pub global: GlobalAdjustments,
    #[serde(default)]
    pub tone_curve: Option<ToneCurve>,
    #[serde(default)]
    pub hsl: BTreeMap<HslColor, HslAdjustment>,
}

impl LocalAdjustmentLayer {
    pub fn radial(
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        radius_y: Option<f32>,
        feather: f32,
    ) -> Self {
        Self {
            name: None,
            opacity: default_local_opacity(),
            mask: MaskShape::Radial {
                center_x,
                center_y,
                radius_x,
                radius_y,
                feather,
                invert: false,
            },
            global: GlobalAdjustments::default(),
            tone_curve: None,
            hsl: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskShape {
    Full,
    Radial {
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        #[serde(default)]
        radius_y: Option<f32>,
        #[serde(default = "default_mask_feather")]
        feather: f32,
        #[serde(default)]
        invert: bool,
    },
    LinearGradient {
        start_x: f32,
        start_y: f32,
        end_x: f32,
        end_y: f32,
        #[serde(default = "default_mask_feather")]
        feather: f32,
        #[serde(default)]
        invert: bool,
    },
    Rectangle {
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        #[serde(default = "default_mask_feather")]
        feather: f32,
        #[serde(default)]
        invert: bool,
    },
    LuminanceRange {
        min: f32,
        max: f32,
        #[serde(default = "default_mask_feather")]
        feather: f32,
        #[serde(default)]
        invert: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhotoEditRecipe {
    #[serde(default = "default_recipe_version")]
    pub version: u32,
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(default)]
    pub global: GlobalAdjustments,
    #[serde(default)]
    pub tone_curve: Option<ToneCurve>,
    #[serde(default)]
    pub hsl: BTreeMap<HslColor, HslAdjustment>,
    #[serde(default, alias = "locals")]
    pub local_adjustments: Vec<LocalAdjustmentLayer>,
}

impl Default for PhotoEditRecipe {
    fn default() -> Self {
        Self {
            version: PHOTO_EDIT_RECIPE_VERSION,
            intent: None,
            global: GlobalAdjustments::default(),
            tone_curve: None,
            hsl: BTreeMap::new(),
            local_adjustments: Vec::new(),
        }
    }
}

impl PhotoEditRecipe {
    pub fn template() -> Self {
        Self {
            tone_curve: Some(ToneCurve::linear()),
            ..Self::default()
        }
    }
}

impl From<BasicAdjustments> for PhotoEditRecipe {
    fn from(value: BasicAdjustments) -> Self {
        Self {
            global: GlobalAdjustments {
                exposure: value.exposure,
                brightness: value.brightness,
                contrast: value.contrast - 1.0,
                saturation: value.saturation - 1.0,
                temperature: value.temperature,
                tint: value.tint,
                hue_shift_degrees: value.hue_shift_degrees,
                ..GlobalAdjustments::default()
            },
            ..Self::default()
        }
    }
}

fn default_recipe_version() -> u32 {
    PHOTO_EDIT_RECIPE_VERSION
}

fn default_local_opacity() -> f32 {
    1.0
}

fn default_mask_feather() -> f32 {
    0.25
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

    #[test]
    fn recipe_template_contains_linear_curve() {
        let recipe = PhotoEditRecipe::template();
        assert_eq!(recipe.version, PHOTO_EDIT_RECIPE_VERSION);
        assert_eq!(
            recipe.tone_curve.expect("tone curve").points,
            ToneCurve::linear().points
        );
    }

    #[test]
    fn recipe_from_basic_adjustments_preserves_legacy_centers() {
        let recipe = PhotoEditRecipe::from(BasicAdjustments {
            exposure: 0.25,
            brightness: 0.1,
            contrast: 1.2,
            saturation: 0.8,
            temperature: -0.05,
            tint: 0.02,
            hue_shift_degrees: 12.0,
        });

        assert_eq!(recipe.global.exposure, 0.25);
        assert_eq!(recipe.global.brightness, 0.1);
        assert!((recipe.global.contrast - 0.2).abs() < f32::EPSILON);
        assert!((recipe.global.saturation + 0.2).abs() < f32::EPSILON);
        assert_eq!(recipe.global.temperature, -0.05);
        assert_eq!(recipe.global.tint, 0.02);
        assert_eq!(recipe.global.hue_shift_degrees, 12.0);
    }

    #[test]
    fn recipe_supports_local_adjustment_alias() {
        let json = r#"
        {
          "version": 1,
          "locals": [
            {
              "opacity": 0.7,
              "mask": {
                "kind": "radial",
                "center_x": 0.5,
                "center_y": 0.5,
                "radius_x": 0.3
              },
              "global": {
                "exposure": 0.2
              }
            }
          ]
        }
        "#;

        let recipe: PhotoEditRecipe = serde_json::from_str(json).expect("recipe json");
        assert_eq!(recipe.local_adjustments.len(), 1);
        assert_eq!(recipe.local_adjustments[0].opacity, 0.7);
        match recipe.local_adjustments[0].mask {
            MaskShape::Radial {
                center_x,
                center_y,
                radius_x,
                radius_y,
                ..
            } => {
                assert_eq!(center_x, 0.5);
                assert_eq!(center_y, 0.5);
                assert_eq!(radius_x, 0.3);
                assert_eq!(radius_y, None);
            }
            _ => panic!("expected radial mask"),
        }
    }
}
