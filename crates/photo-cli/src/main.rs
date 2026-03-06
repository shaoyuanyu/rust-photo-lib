//! 照片处理命令行工具入口。

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use photo_core::ImageFrame;
use photo_core::{BasicAdjustments, Pipeline};
use photo_imageops::{BasicAdjustStage, load_image, save_image};
use photo_models::StyleTransferModel;

/// 命令行参数定义。
#[derive(Parser, Debug)]
#[command(author, version, about = "Rust photo processing CLI")]
struct Cli {
    /// 输入图像路径。
    #[arg(long)]
    input: PathBuf,
    /// 输出图像路径。
    #[arg(long)]
    output: PathBuf,

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

    /// 风格迁移模型路径（ONNX）。
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

    let mut image = load_image(&cli.input).map_err(anyhow::Error::msg)?;

    let mut pipeline = Pipeline::new();
    pipeline.push(BasicAdjustStage {
        params: BasicAdjustments {
            exposure: cli.exposure,
            brightness: cli.brightness,
            contrast: cli.contrast,
            saturation: cli.saturation,
            temperature: cli.temperature,
            tint: cli.tint,
            hue_shift_degrees: cli.hue_shift_degrees,
        },
    });

    image = pipeline.run(image).map_err(anyhow::Error::msg)?;

    if let Some(model_path) = &cli.style_model {
        let model = StyleTransferModel::new(model_path);
        image = apply_style_transfer(cli.backend, &model, &image)?;
    }

    save_image(&image, &cli.output)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to save image to {}", cli.output.display()))?;

    Ok(())
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
