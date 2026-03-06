//! 图像读写与基础处理工具。

use std::path::Path;

use image::{ImageBuffer, Rgb, RgbImage};
use photo_core::{BasicAdjustments, ImageFrame, PhotoError, Result, Stage};

/// 张量转换的归一化选项。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorNormalization {
    /// 归一化到 0..1。
    ZeroOne,
    /// 归一化到 -1..1。
    MinusOneOne,
    /// 保持 0..255。
    ZeroTwoFiftyFive,
}

/// 应用基础调节的管线阶段。
#[derive(Debug, Clone, Copy)]
pub struct BasicAdjustStage {
    /// 调整参数。
    pub params: BasicAdjustments,
}

impl Stage for BasicAdjustStage {
    /// 阶段名称。
    fn name(&self) -> &'static str {
        "basic-adjust"
    }

    /// 对输入帧应用基础调节。
    fn apply(&self, input: ImageFrame) -> Result<ImageFrame> {
        apply_basic_adjustments(input, self.params)
    }
}

/// 从磁盘加载图像并转换为 RGB 帧。
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<ImageFrame> {
    let img = image::open(path)
        .map_err(|err| PhotoError::InvalidImageData(err.to_string()))?
        .to_rgb8();
    ImageFrame::new(img.width(), img.height(), img.into_raw())
}

/// 将 RGB 帧保存到磁盘。
pub fn save_image<P: AsRef<Path>>(frame: &ImageFrame, path: P) -> Result<()> {
    let image = frame_to_rgb_image(frame)?;
    image
        .save(path)
        .map_err(|err| PhotoError::InvalidImageData(err.to_string()))
}

/// 将 RGB 帧缩放到指定尺寸。
pub fn resize_rgb(frame: &ImageFrame, target_width: u32, target_height: u32) -> Result<ImageFrame> {
    let image = frame_to_rgb_image(frame)?;
    let resized = image::imageops::resize(
        &image,
        target_width,
        target_height,
        image::imageops::FilterType::Triangle,
    );
    ImageFrame::new(target_width, target_height, resized.into_raw())
}

/// 计算受最大边限制并对齐到指定倍数的尺寸。
pub fn fit_within_max_and_multiple(
    width: u32,
    height: u32,
    max_dim: u32,
    multiple: u32,
) -> (u32, u32) {
    let mut target_w = width;
    let mut target_h = height;

    if max_dim > 0 && (target_w > max_dim || target_h > max_dim) {
        let ratio = max_dim as f32 / target_w.max(target_h) as f32;
        target_w = (target_w as f32 * ratio).floor() as u32;
        target_h = (target_h as f32 * ratio).floor() as u32;
    }

    if multiple > 1 {
        let snap = |value: u32| {
            let snapped = (value / multiple) * multiple;
            if snapped == 0 { multiple } else { snapped }
        };
        target_w = snap(target_w);
        target_h = snap(target_h);
    }

    (target_w.max(1), target_h.max(1))
}

/// 应用曝光/亮度/对比度/饱和度/色温/色调/色相调节。
pub fn apply_basic_adjustments(input: ImageFrame, params: BasicAdjustments) -> Result<ImageFrame> {
    let mut out = input.data.clone();
    let exposure_mul = 2.0f32.powf(params.exposure);

    for chunk in out.chunks_exact_mut(3) {
        let mut r = chunk[0] as f32 / 255.0;
        let mut g = chunk[1] as f32 / 255.0;
        let mut b = chunk[2] as f32 / 255.0;

        r *= exposure_mul;
        g *= exposure_mul;
        b *= exposure_mul;

        r += params.brightness;
        g += params.brightness;
        b += params.brightness;

        r = ((r - 0.5) * params.contrast) + 0.5;
        g = ((g - 0.5) * params.contrast) + 0.5;
        b = ((b - 0.5) * params.contrast) + 0.5;

        r += params.temperature * 0.08;
        b -= params.temperature * 0.08;
        g += params.tint * 0.05;

        let (mut h, mut s, v) = rgb_to_hsv(r, g, b);
        h += params.hue_shift_degrees;
        if h < 0.0 {
            h += 360.0;
        }
        if h >= 360.0 {
            h -= 360.0;
        }
        s *= params.saturation;
        let (nr, ng, nb) = hsv_to_rgb(h, s, v);

        chunk[0] = to_u8(nr);
        chunk[1] = to_u8(ng);
        chunk[2] = to_u8(nb);
    }

    ImageFrame::new(input.width, input.height, out)
}

/// 将 RGB 帧转为 NCHW f32（0..1 或 -1..1 归一化）。
pub fn image_to_nchw_f32(frame: &ImageFrame, normalize_minus1_1: bool) -> Vec<f32> {
    let normalization = if normalize_minus1_1 {
        TensorNormalization::MinusOneOne
    } else {
        TensorNormalization::ZeroOne
    };
    image_to_nchw_f32_with_normalization(frame, normalization)
}

/// 将 RGB 帧转为 NCHW f32，并指定归一化模式。
pub fn image_to_nchw_f32_with_normalization(
    frame: &ImageFrame,
    normalization: TensorNormalization,
) -> Vec<f32> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let plane = width * height;
    let mut out = vec![0.0; plane * 3];

    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) * 3;
            let idx = y * width + x;

            let mut r = frame.data[pixel_index] as f32;
            let mut g = frame.data[pixel_index + 1] as f32;
            let mut b = frame.data[pixel_index + 2] as f32;

            match normalization {
                TensorNormalization::ZeroOne => {
                    r /= 255.0;
                    g /= 255.0;
                    b /= 255.0;
                }
                TensorNormalization::MinusOneOne => {
                    r = (r / 255.0) * 2.0 - 1.0;
                    g = (g / 255.0) * 2.0 - 1.0;
                    b = (b / 255.0) * 2.0 - 1.0;
                }
                TensorNormalization::ZeroTwoFiftyFive => {}
            }

            out[idx] = r;
            out[plane + idx] = g;
            out[plane * 2 + idx] = b;
        }
    }

    out
}

/// 将 NCHW f32 数据转为 RGB 帧（0..1 或 -1..1 归一化）。
pub fn nchw_f32_to_image(
    width: u32,
    height: u32,
    data: &[f32],
    denormalize_minus1_1: bool,
) -> Result<ImageFrame> {
    let normalization = if denormalize_minus1_1 {
        TensorNormalization::MinusOneOne
    } else {
        TensorNormalization::ZeroOne
    };
    nchw_f32_to_image_with_normalization(width, height, data, normalization)
}

/// 将 NCHW f32 数据转为 RGB 帧，并指定归一化模式。
pub fn nchw_f32_to_image_with_normalization(
    width: u32,
    height: u32,
    data: &[f32],
    normalization: TensorNormalization,
) -> Result<ImageFrame> {
    let width = width as usize;
    let height = height as usize;
    let plane = width * height;
    if data.len() != plane * 3 {
        return Err(PhotoError::Model(format!(
            "unexpected tensor length: {}, expected {}",
            data.len(),
            plane * 3
        )));
    }

    let mut out = vec![0u8; plane * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let pixel = idx * 3;

            let r = data[idx];
            let g = data[plane + idx];
            let b = data[plane * 2 + idx];

            match normalization {
                TensorNormalization::ZeroOne => {
                    out[pixel] = to_u8(r);
                    out[pixel + 1] = to_u8(g);
                    out[pixel + 2] = to_u8(b);
                }
                TensorNormalization::MinusOneOne => {
                    out[pixel] = to_u8((r + 1.0) * 0.5);
                    out[pixel + 1] = to_u8((g + 1.0) * 0.5);
                    out[pixel + 2] = to_u8((b + 1.0) * 0.5);
                }
                TensorNormalization::ZeroTwoFiftyFive => {
                    out[pixel] = r.clamp(0.0, 255.0).round() as u8;
                    out[pixel + 1] = g.clamp(0.0, 255.0).round() as u8;
                    out[pixel + 2] = b.clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    }

    ImageFrame::new(width as u32, height as u32, out)
}

/// 从帧构建 `RgbImage`。
fn frame_to_rgb_image(frame: &ImageFrame) -> Result<RgbImage> {
    ImageBuffer::<Rgb<u8>, _>::from_raw(frame.width, frame.height, frame.data.clone())
        .ok_or_else(|| PhotoError::InvalidImageData("failed to create image buffer".into()))
}

/// 将 0..1 的浮点值钳制到字节。
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// RGB 转 HSV（各分量在 0..1）。
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
}

/// HSV 转 RGB（各分量在 0..1）。
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0);
    let s = s.clamp(0.0, 4.0);
    let v = v.clamp(0.0, 1.0);

    let c = v * s;
    let x = c * (1.0 - (((h / 60.0) % 2.0) - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match h {
        h if (0.0..60.0).contains(&h) => (c, x, 0.0),
        h if (60.0..120.0).contains(&h) => (x, c, 0.0),
        h if (120.0..180.0).contains(&h) => (0.0, c, x),
        h if (180.0..240.0).contains(&h) => (0.0, x, c),
        h if (240.0..300.0).contains(&h) => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r1 + m, g1 + m, b1 + m)
}

#[cfg(test)]
mod tests {
    use photo_core::BasicAdjustments;

    use super::*;

    /// 验证基础调节保持尺寸与缓冲长度。
    #[test]
    fn basic_adjust_keeps_shape() {
        let input = ImageFrame::new(2, 1, vec![10, 20, 30, 200, 100, 50]).expect("valid frame");
        let out =
            apply_basic_adjustments(input.clone(), BasicAdjustments::default()).expect("adjust ok");
        assert_eq!(out.width, input.width);
        assert_eq!(out.height, input.height);
        assert_eq!(out.data.len(), input.data.len());
    }

    /// 0..1 归一化下的 NCHW 往返转换。
    #[test]
    fn nchw_roundtrip_zero_one() {
        let input = ImageFrame::new(2, 1, vec![0, 128, 255, 255, 64, 0]).expect("valid frame");
        let tensor = image_to_nchw_f32(&input, false);
        let out = nchw_f32_to_image(2, 1, &tensor, false).expect("roundtrip ok");
        assert_eq!(out.data, input.data);
    }

    /// -1..1 归一化下的 NCHW 往返转换。
    #[test]
    fn nchw_roundtrip_minus_one_one() {
        let input = ImageFrame::new(1, 1, vec![32, 128, 224]).expect("valid frame");
        let tensor = image_to_nchw_f32(&input, true);
        let out = nchw_f32_to_image(1, 1, &tensor, true).expect("roundtrip ok");
        assert_eq!(out.data, input.data);
    }

    /// 0..255 归一化下的 NCHW 往返转换。
    #[test]
    fn nchw_roundtrip_zero_two_fifty_five() {
        let input = ImageFrame::new(1, 2, vec![12, 34, 56, 200, 150, 100]).expect("valid frame");
        let tensor = image_to_nchw_f32_with_normalization(&input, TensorNormalization::ZeroTwoFiftyFive);
        let out = nchw_f32_to_image_with_normalization(
            1,
            2,
            &tensor,
            TensorNormalization::ZeroTwoFiftyFive,
        )
        .expect("roundtrip ok");
        assert_eq!(out.data, input.data);
    }

    /// 验证缩放策略会缩小并对齐倍数。
    #[test]
    fn fit_within_max_and_multiple_scales_down() {
        let (w, h) = fit_within_max_and_multiple(1920, 1080, 800, 8);
        assert_eq!((w, h), (800, 448));
    }
}
