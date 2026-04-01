//! 照片处理命令行工具入口。

use std::{fs, path::{Path, PathBuf}};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use photo_core::{BeautySettings, GlobalAdjustments, ImageFrame, PhotoEditRecipe, Pipeline};
use photo_imageops::{PhotoEditStage, apply_beauty_filters, load_image, save_image};
use photo_models::StyleTransferModel;
use photo_onnx::OnnxEngine;
#[cfg(feature = "ort-backend")]
use photo_models::{FaceBeautyProcessor, FaceDetectorModel, FaceLandmarkModel};

/// 命令行参数定义。
#[derive(Parser, Debug)]
#[command(author, version, about = "Rust photo processing CLI")]
struct Cli {
    /// 输入图像路径。
    #[arg(long, required_unless_present = "dump_recipe_template")]
    input: Option<PathBuf>,
    /// 输出图像路径。
    #[arg(long, required_unless_present = "dump_recipe_template")]
    output: Option<PathBuf>,

    /// 曝光（EV）。
    #[arg(long, default_value_t = 0.0)]
    exposure: f32,
    /// 亮度偏移。
    #[arg(long, default_value_t = 0.0)]
    brightness: f32,
    /// 对比度倍数。
    #[arg(long, default_value_t = 1.0)]
    contrast: f32,
    /// 饱和度倍数。
    #[arg(long, default_value_t = 1.0)]
    saturation: f32,
    /// 色温偏移。
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// 色调偏移。
    #[arg(long, default_value_t = 0.0)]
    tint: f32,
    /// 色相偏移（度）。
    #[arg(long = "hue", default_value_t = 0.0)]
    hue_shift_degrees: f32,

    /// 编辑配方路径（JSON）。
    #[arg(long)]
    recipe: Option<PathBuf>,
    #[arg(long)]
    dump_recipe_template: bool,

    #[arg(long)]
    style_model: Option<PathBuf>,
    #[arg(long)]
    face_det_model: Option<PathBuf>,
    #[arg(long)]
    face_landmark_model: Option<PathBuf>,
    /// 推理后端选择。
    #[arg(long, value_enum, default_value_t = Backend::Ort)]
    backend: Backend,
}

/// 推理后端枚举。
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Backend {
    /// Tract 后端。
    Tract,
    /// ONNX Runtime 后端。
    Ort,
}

/// CLI 入口。
fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.dump_recipe_template {
        println!(
            "{}",
            serde_json::to_string_pretty(&PhotoEditRecipe::template())
                .context("failed to serialize recipe template")?
        );
        return Ok(());
    }

    let input_path = cli
        .input
        .clone()
        .context("`--input` is required unless `--dump-recipe-template` is used")?;
    let output_path = cli
        .output
        .clone()
        .context("`--output` is required unless `--dump-recipe-template` is used")?;
    let recipe = load_effective_recipe(&cli)?;

    let mut image = load_image(&input_path).map_err(anyhow::Error::msg)?;

    let mut pipeline = Pipeline::new();
    pipeline.push(PhotoEditStage {
        recipe: recipe.clone(),
    });

    image = pipeline.run(image).map_err(anyhow::Error::msg)?;
    image = apply_face_beauty(&cli, &recipe, &image)?;

    if let Some(model_path) = &cli.style_model {
        let model = StyleTransferModel::new(model_path);
        image = match cli.backend {
            Backend::Tract => apply_style_transfer::<photo_backend_tract::TractEngine>(cli.backend, &model, &image)?,
            Backend::Ort => apply_style_transfer::<photo_backend_ort::OrtEngine>(cli.backend, &model, &image)?,
        };
    }

    save_image(&image, &output_path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to save image to {}", output_path.display()))?;

    Ok(())
}

fn apply_face_beauty(
    cli: &Cli,
    recipe: &PhotoEditRecipe,
    image: &ImageFrame,
) -> Result<ImageFrame> {
    let beauty = recipe.beauty.clamped();
    if beauty.is_identity() {
        return Ok(image.clone());
    }

    if !beauty.needs_face_detection() {
        return apply_beauty_filters(image.clone(), beauty).map_err(anyhow::Error::msg);
    }

    if !matches!(cli.backend, Backend::Ort) {
        return Err(anyhow::anyhow!(
            "face reshape only supports the `ort` backend; rerun with `--backend ort`"
        ));
    }

    let det_model = cli.face_det_model.as_ref().with_context(|| {
        "face reshape requires `--face-det-model` when `beauty.thin_face` or `beauty.big_eye` is enabled"
    })?;
    let landmark_model = cli.face_landmark_model.as_ref().with_context(|| {
        "face reshape requires `--face-landmark-model` when `beauty.thin_face` or `beauty.big_eye` is enabled"
    })?;

    run_ort_face_beauty(&beauty, det_model, landmark_model, image)
}

fn load_effective_recipe(cli: &Cli) -> Result<PhotoEditRecipe> {
    let mut recipe = if let Some(path) = &cli.recipe {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read recipe file {}", path.display()))?;
        serde_json::from_str::<PhotoEditRecipe>(&raw)
            .with_context(|| format!("failed to parse recipe file {}", path.display()))?
    } else {
        PhotoEditRecipe::default()
    };

    apply_cli_global_overrides(&mut recipe.global, cli);
    Ok(recipe)
}

fn apply_cli_global_overrides(global: &mut GlobalAdjustments, cli: &Cli) {
    set_if_non_default(&mut global.exposure, cli.exposure, 0.0);
    set_if_non_default(&mut global.brightness, cli.brightness, 0.0);
    set_if_non_default(&mut global.contrast, cli.contrast - 1.0, 0.0);
    set_if_non_default(&mut global.saturation, cli.saturation - 1.0, 0.0);
    set_if_non_default(&mut global.temperature, cli.temperature, 0.0);
    set_if_non_default(&mut global.tint, cli.tint, 0.0);
    set_if_non_default(&mut global.hue_shift_degrees, cli.hue_shift_degrees, 0.0);
}

fn set_if_non_default(target: &mut f32, candidate: f32, default: f32) {
    if (candidate - default).abs() > f32::EPSILON {
        *target = candidate;
    }
}

/// 根据后端选择执行风格迁移。
fn apply_style_transfer<E: OnnxEngine + Default>(_: Backend, model: &StyleTransferModel, image: &ImageFrame) -> Result<ImageFrame> {
    let mut engine = E::default();
    model.run(&mut engine, image).map_err(anyhow::Error::msg)
}

#[cfg(feature = "ort-backend")]
fn run_ort_face_beauty(
    beauty: &BeautySettings,
    det_model: &Path,
    landmark_model: &Path,
    image: &ImageFrame,
) -> Result<ImageFrame> {
    let processor = FaceBeautyProcessor::new(
        FaceDetectorModel::new(det_model),
        FaceLandmarkModel::new(landmark_model),
    );
    let mut detector_engine = photo_backend_ort::OrtEngine::default();
    let mut landmark_engine = photo_backend_ort::OrtEngine::default();
    processor
        .process(&mut detector_engine, &mut landmark_engine, image, *beauty)
        .map_err(anyhow::Error::msg)
}

#[cfg(not(feature = "ort-backend"))]
fn run_ort_face_beauty(
    _beauty: &BeautySettings,
    _det_model: &Path,
    _landmark_model: &Path,
    _image: &ImageFrame,
) -> Result<ImageFrame> {
    Err(anyhow::anyhow!(
        "face reshape requires the ORT backend. Rebuild with `--features ort-backend`"
    ))
}

/// 使用 Tract 后端执行风格迁移（需启用特性）。
#[cfg(feature = "tract-backend")]
fn run_tract_style_transfer(model: &StyleTransferModel, image: &ImageFrame) -> Result<ImageFrame> {
    let mut engine = photo_backend_tract::TractEngine::default();
    model.run(&mut engine, image).map_err(anyhow::Error::msg)
}
