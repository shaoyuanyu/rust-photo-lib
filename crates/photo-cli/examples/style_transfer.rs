use std::path::PathBuf;

use photo_core::{BasicAdjustments, Pipeline};
use photo_imageops::{BasicAdjustStage, load_image, save_image};
use photo_models::StyleTransferModel;

fn main() {
    let input = PathBuf::from("input.jpg");
    let output = PathBuf::from("output_style.jpg");
    let model_path = PathBuf::from("style.onnx");

    let image = load_image(&input).expect("load image");

    let mut pipeline = Pipeline::new();
    pipeline.push(BasicAdjustStage {
        params: BasicAdjustments {
            contrast: 1.05,
            saturation: 1.1,
            ..Default::default()
        },
    });
    let toned = pipeline.run(image).expect("run pipeline");

    #[cfg(feature = "tract-backend")]
    {
        let mut engine = photo_backend_tract::TractEngine::default();
        let model = StyleTransferModel::new(model_path);
        let stylized = model.run(&mut engine, &toned).expect("run style transfer");
        save_image(&stylized, &output).expect("save image");
    }

    #[cfg(not(feature = "tract-backend"))]
    {
        let _ = model_path;
        let _ = toned;
        eprintln!(
            "Enable tract backend first: cargo run -p photo-cli --features tract-backend --example style_transfer"
        );
    }
}
