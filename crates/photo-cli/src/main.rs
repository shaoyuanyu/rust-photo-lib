use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use photo_core::ImageFrame;
use photo_core::{BasicAdjustments, Pipeline};
use photo_imageops::{BasicAdjustStage, load_image, save_image};
use photo_models::StyleTransferModel;

#[derive(Parser, Debug)]
#[command(author, version, about = "Rust photo processing CLI")]
struct Cli {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 0.0)]
    exposure: f32,
    #[arg(long, default_value_t = 0.0)]
    brightness: f32,
    #[arg(long, default_value_t = 1.0)]
    contrast: f32,
    #[arg(long, default_value_t = 1.0)]
    saturation: f32,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0.0)]
    tint: f32,
    #[arg(long = "hue", default_value_t = 0.0)]
    hue_shift_degrees: f32,

    #[arg(long)]
    style_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Tract)]
    backend: Backend,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Backend {
    Tract,
    Ort,
}

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

#[cfg(feature = "tract-backend")]
fn run_tract_style_transfer(model: &StyleTransferModel, image: &ImageFrame) -> Result<ImageFrame> {
    let mut engine = photo_backend_tract::TractEngine::default();
    model.run(&mut engine, image).map_err(anyhow::Error::msg)
}

#[cfg(not(feature = "tract-backend"))]
fn run_tract_style_transfer(
    _model: &StyleTransferModel,
    _image: &ImageFrame,
) -> Result<ImageFrame> {
    Err(anyhow::anyhow!(
        "tract backend is not enabled. Rebuild with `--features tract-backend`"
    ))
}

#[cfg(feature = "ort-backend")]
fn run_ort_style_transfer(model: &StyleTransferModel, image: &ImageFrame) -> Result<ImageFrame> {
    let mut engine = photo_backend_ort::OrtEngine::default();
    model.run(&mut engine, image).map_err(anyhow::Error::msg)
}

#[cfg(not(feature = "ort-backend"))]
fn run_ort_style_transfer(_model: &StyleTransferModel, _image: &ImageFrame) -> Result<ImageFrame> {
    Err(anyhow::anyhow!(
        "ort backend is not enabled. Rebuild with `--features ort-backend`"
    ))
}
