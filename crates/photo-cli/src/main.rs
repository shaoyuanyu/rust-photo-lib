//! 照片处理命令行工具入口。

use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use photo_core::{GlobalAdjustments, ImageFrame, PhotoEditRecipe, Pipeline};
use photo_imageops::{PhotoEditStage, load_image, save_image};
use photo_models::StyleTransferModel;

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
    pipeline.push(PhotoEditStage { recipe });

    image = pipeline.run(image).map_err(anyhow::Error::msg)?;

    if let Some(model_path) = &cli.style_model {
        let model = StyleTransferModel::new(model_path);
        image = apply_style_transfer(cli.backend, &model, &image)?;
    }

    save_image(&image, &output_path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to save image to {}", output_path.display()))?;

    Ok(())
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
fn apply_style_transfer(
    backend: Backend,
    model: &StyleTransferModel,
    image: &ImageFrame,
) -> Result<ImageFrame> {
    match backend {
        Backend::Tract => run_tract_style_transfer(model, image),
        Backend::Ort => run_ort_style_transfer(model, image),
    }
}

/// 使用 Tract 后端执行风格迁移（需启用特性）。
#[cfg(feature = "tract-backend")]
fn run_tract_style_transfer(model: &StyleTransferModel, image: &ImageFrame) -> Result<ImageFrame> {
    let mut engine = photo_backend_tract::TractEngine::default();
    model.run(&mut engine, image).map_err(anyhow::Error::msg)
}

/// Tract 后端未启用时的报错实现。
#[cfg(not(feature = "tract-backend"))]
fn run_tract_style_transfer(
    _model: &StyleTransferModel,
    _image: &ImageFrame,
) -> Result<ImageFrame> {
    Err(anyhow::anyhow!(
        "tract backend is not enabled. Rebuild with `--features tract-backend`"
    ))
}

/// 使用 ORT 后端执行风格迁移（需启用特性）。
#[cfg(feature = "ort-backend")]
fn run_ort_style_transfer(model: &StyleTransferModel, image: &ImageFrame) -> Result<ImageFrame> {
    let mut engine = photo_backend_ort::OrtEngine::default();
    model.run(&mut engine, image).map_err(anyhow::Error::msg)
}

/// ORT 后端未启用时的报错实现。
#[cfg(not(feature = "ort-backend"))]
fn run_ort_style_transfer(_model: &StyleTransferModel, _image: &ImageFrame) -> Result<ImageFrame> {
    Err(anyhow::anyhow!(
        "ort backend is not enabled. Rebuild with `--features ort-backend`"
    ))
}
