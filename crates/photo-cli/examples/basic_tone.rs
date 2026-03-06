//! 基础调色示例。

use std::path::PathBuf;

use photo_core::{BasicAdjustments, Pipeline};
use photo_imageops::{BasicAdjustStage, load_image, save_image};

/// 示例入口：应用基础调色并保存输出。
fn main() {
    let input = PathBuf::from("input.jpg");
    let output = PathBuf::from("output_tone.jpg");

    let image = load_image(&input).expect("load image");

    let mut pipeline = Pipeline::new();
    pipeline.push(BasicAdjustStage {
        params: BasicAdjustments {
            exposure: 0.2,
            brightness: 0.03,
            contrast: 1.1,
            saturation: 1.15,
            temperature: 0.05,
            tint: 0.02,
            hue_shift_degrees: 2.0,
        },
    });

    let out = pipeline.run(image).expect("run pipeline");
    save_image(&out, &output).expect("save image");
}
