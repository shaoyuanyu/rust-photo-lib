//! 图像读写与基础处理工具。

use std::{collections::BTreeMap, path::Path};

use image::{ImageBuffer, Rgb, RgbImage};
use photo_core::{
    BasicAdjustments, GlobalAdjustments, HslAdjustment, HslColor, ImageFrame, LocalAdjustmentLayer,
    MaskShape, PhotoEditRecipe, PhotoError, Result, Stage, ToneCurve,
};

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

#[derive(Debug, Clone)]
/// 应用编辑配方的管线阶段。
pub struct PhotoEditStage {
    pub recipe: PhotoEditRecipe,
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

impl Stage for PhotoEditStage {
    fn name(&self) -> &'static str {
        "photo-edit"
    }

    fn apply(&self, input: ImageFrame) -> Result<ImageFrame> {
        apply_photo_edit_recipe(input, &self.recipe)
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
    apply_global_adjustments(input, GlobalAdjustments::from(params))
}

/// 根据照片编辑配方执行全局与局部调整。
pub fn apply_photo_edit_recipe(input: ImageFrame, recipe: &PhotoEditRecipe) -> Result<ImageFrame> {
    // 按照全局 → 曲线 → HSL → 局部的顺序执行配方。
    let mut frame = apply_edit_stack(
        input,
        recipe.global,
        recipe.tone_curve.as_ref(),
        &recipe.hsl,
    )?;
    for layer in &recipe.local_adjustments {
        frame = apply_local_adjustment_layer(frame, layer)?;
    }
    Ok(frame)
}

fn apply_edit_stack(
    input: ImageFrame,
    global: GlobalAdjustments,
    tone_curve: Option<&ToneCurve>,
    hsl: &BTreeMap<HslColor, HslAdjustment>,
) -> Result<ImageFrame> {
    // 先应用全局调整，再叠加曲线与 HSL。
    let mut frame = apply_global_adjustments(input, global)?;
    if let Some(curve) = tone_curve {
        frame = apply_tone_curve(frame, curve)?;
    }
    if !hsl.is_empty() {
        frame = apply_hsl_adjustments(frame, hsl)?;
    }
    Ok(frame)
}

/// 应用曝光/对比度/色彩等全局调节。
pub fn apply_global_adjustments(
    input: ImageFrame,
    params: GlobalAdjustments,
) -> Result<ImageFrame> {
    let mut out = input.data.clone();
    let exposure_mul = 2.0f32.powf(params.exposure);
    let contrast_mul = (1.0 + params.contrast).max(0.0);
    let saturation_mul = (1.0 + params.saturation).max(0.0);

    for chunk in out.chunks_exact_mut(3) {
        // RGB 归一化到 0..1，便于在 HSV 空间调整。
        let mut r = chunk[0] as f32 / 255.0;
        let mut g = chunk[1] as f32 / 255.0;
        let mut b = chunk[2] as f32 / 255.0;

        r *= exposure_mul;
        g *= exposure_mul;
        b *= exposure_mul;

        r += params.brightness;
        g += params.brightness;
        b += params.brightness;

        r += params.temperature * 0.08;
        b -= params.temperature * 0.08;
        g += params.tint * 0.05;

        let (mut h, mut s, mut v) = rgb_to_hsv(r, g, b);
        let luma = rgb_luma(r, g, b);
        // 亮度控制基于 luma 进行区域分配后，再做对比度缩放。
        v = apply_tone_controls(v, luma, params, contrast_mul);
        h = (h + params.hue_shift_degrees).rem_euclid(360.0);
        s *= saturation_mul;
        // 自然饱和度优先提升低饱和像素，并保护肤色与高光。
        s = apply_vibrance(s, params.vibrance, h, v);
        let (nr, ng, nb) = hsv_to_rgb(h, s, v);

        chunk[0] = to_u8(nr);
        chunk[1] = to_u8(ng);
        chunk[2] = to_u8(nb);
    }

    ImageFrame::new(input.width, input.height, out)
}

/// 使用曲线对每个通道进行分段线性映射。
pub fn apply_tone_curve(input: ImageFrame, curve: &ToneCurve) -> Result<ImageFrame> {
    // 预处理曲线点以确保端点存在且按 x 排序。
    let curve_points = prepare_curve_points(curve);
    let mut out = input.data.clone();

    for value in &mut out {
        let normalized = *value as f32 / 255.0;
        // 对每个通道执行分段线性插值。
        *value = to_u8(evaluate_curve(&curve_points, normalized));
    }

    ImageFrame::new(input.width, input.height, out)
}

/// 在 HSV 空间按色相区域应用 HSL 调整。
pub fn apply_hsl_adjustments(
    input: ImageFrame,
    adjustments: &BTreeMap<HslColor, HslAdjustment>,
) -> Result<ImageFrame> {
    let mut out = input.data.clone();

    for chunk in out.chunks_exact_mut(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let (mut h, mut s, mut v) = rgb_to_hsv(r, g, b);

        let mut hue_delta = 0.0;
        let mut saturation_delta = 0.0;
        let mut luminance_delta = 0.0;
        // 依据色相中心做加权混合，避免色块边界突兀。
        for (color, adjustment) in adjustments {
            let weight = hue_weight(h, hsl_color_center(*color), 60.0);
            hue_delta += adjustment.hue * 45.0 * weight;
            saturation_delta += adjustment.saturation * weight;
            luminance_delta += adjustment.luminance * 0.5 * weight;
        }

        h += hue_delta;
        s *= (1.0 + saturation_delta).max(0.0);
        v = (v + luminance_delta).clamp(0.0, 1.0);

        let (nr, ng, nb) = hsv_to_rgb(h, s, v);
        chunk[0] = to_u8(nr);
        chunk[1] = to_u8(ng);
        chunk[2] = to_u8(nb);
    }

    ImageFrame::new(input.width, input.height, out)
}

/// 应用单个局部调整层并按遮罩混合输出。
pub fn apply_local_adjustment_layer(
    input: ImageFrame,
    layer: &LocalAdjustmentLayer,
) -> Result<ImageFrame> {
    // 每个局部层都在完整图像上生成结果，再用遮罩混合。
    let edited = apply_edit_stack(
        input.clone(),
        layer.global,
        layer.tone_curve.as_ref(),
        &layer.hsl,
    )?;
    blend_with_mask(&input, &edited, &layer.mask, layer.opacity)
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
    // NCHW 需要 3 个平面并按 [1, 3, H, W] 展开。
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

fn apply_tone_controls(value: f32, luma: f32, params: GlobalAdjustments, contrast_mul: f32) -> f32 {
    // 根据不同亮度区间累计调整，然后统一应用对比度。
    let with_tonal = (value + tonal_region_delta(params, luma)).clamp(0.0, 1.0);
    apply_contrast(with_tonal, contrast_mul)
}

fn tonal_region_delta(params: GlobalAdjustments, luma: f32) -> f32 {
    let shadows = shadow_mask(luma);
    let highlights = highlight_mask(luma);
    let blacks = black_mask(luma);
    let whites = white_mask(luma);

    tone_zone_delta(luma, params.shadows, shadows, 0.85)
        + tone_zone_delta(luma, params.highlights, highlights, 0.65)
        + tone_zone_delta(luma, params.blacks, blacks, 0.75)
        + tone_zone_delta(luma, params.whites, whites, 0.55)
}

fn tone_zone_delta(luma: f32, amount: f32, mask: f32, strength: f32) -> f32 {
    let amount = amount.clamp(-1.0, 1.0);
    if amount >= 0.0 {
        amount * mask * (1.0 - luma) * strength
    } else {
        amount * mask * (0.25 + luma * 0.75) * strength
    }
}

fn apply_contrast(value: f32, contrast_mul: f32) -> f32 {
    ((value - 0.5) * contrast_mul + 0.5).clamp(0.0, 1.0)
}

fn shadow_mask(luma: f32) -> f32 {
    let luma = luma.clamp(0.0, 1.0);
    (1.0 - smoothstep(0.18, 0.72, luma)) * (1.0 - black_mask(luma) * 0.35)
}

fn highlight_mask(luma: f32) -> f32 {
    let luma = luma.clamp(0.0, 1.0);
    smoothstep(0.28, 0.82, luma) * (1.0 - white_mask(luma) * 0.35)
}

fn black_mask(luma: f32) -> f32 {
    1.0 - smoothstep(0.0, 0.28, luma)
}

fn white_mask(luma: f32) -> f32 {
    smoothstep(0.72, 1.0, luma)
}

fn apply_vibrance(saturation: f32, vibrance: f32, hue: f32, value: f32) -> f32 {
    let saturation = saturation.clamp(0.0, 4.0);
    let vibrance = vibrance.clamp(-1.0, 1.0);
    let low_sat_bias = (1.0 - saturation.clamp(0.0, 1.0)).powf(1.35);
    let skin_protection =
        1.0 - 0.35 * hue_weight(hue, 28.0, 55.0) * smoothstep(0.08, 0.75, saturation);
    let highlight_protection = 1.0 - 0.20 * smoothstep(0.75, 1.0, value);

    if vibrance >= 0.0 {
        // 正向振动优先提升低饱和区域。
        (saturation + vibrance * low_sat_bias * skin_protection * highlight_protection * 0.9)
            .clamp(0.0, 4.0)
    } else {
        // 负向振动更温和地回落高饱和区域。
        let negative_rolloff = 0.55 + 0.45 * (1.0 - low_sat_bias);
        (saturation * (1.0 + vibrance * negative_rolloff)).clamp(0.0, 4.0)
    }
}

fn blend_with_mask(
    base: &ImageFrame,
    edited: &ImageFrame,
    mask: &MaskShape,
    opacity: f32,
) -> Result<ImageFrame> {
    if base.width != edited.width || base.height != edited.height {
        return Err(PhotoError::Pipeline(
            "local adjustment layers require matching image sizes".into(),
        ));
    }

    let opacity = opacity.clamp(0.0, 1.0);
    let mut out = base.data.clone();
    let width = base.width.max(1);
    let height = base.height.max(1);

    for y in 0..base.height as usize {
        for x in 0..base.width as usize {
            let pixel = (y * base.width as usize + x) * 3;
            let context = PixelContext::new(base, x, y, width, height);
            let weight = opacity * evaluate_mask(mask, &context);

            // 按遮罩权重在原图与编辑图之间线性混合。
            out[pixel] = blend_channel(base.data[pixel], edited.data[pixel], weight);
            out[pixel + 1] = blend_channel(base.data[pixel + 1], edited.data[pixel + 1], weight);
            out[pixel + 2] = blend_channel(base.data[pixel + 2], edited.data[pixel + 2], weight);
        }
    }

    ImageFrame::new(base.width, base.height, out)
}

#[derive(Clone, Copy)]
struct PixelContext {
    x: f32,
    y: f32,
    luma: f32,
}

impl PixelContext {
    fn new(frame: &ImageFrame, x: usize, y: usize, width: u32, height: u32) -> Self {
        let pixel = (y * frame.width as usize + x) * 3;
        let r = frame.data[pixel] as f32 / 255.0;
        let g = frame.data[pixel + 1] as f32 / 255.0;
        let b = frame.data[pixel + 2] as f32 / 255.0;

        Self {
            // 使用像素中心位置并归一化到 0..1。
            x: (x as f32 + 0.5) / width as f32,
            y: (y as f32 + 0.5) / height as f32,
            luma: rgb_luma(r, g, b),
        }
    }
}

fn evaluate_mask(mask: &MaskShape, context: &PixelContext) -> f32 {
    match mask {
        MaskShape::Full => 1.0,
        MaskShape::Radial {
            center_x,
            center_y,
            radius_x,
            radius_y,
            feather,
            invert,
        } => apply_invert(
            radial_mask(
                context.x,
                context.y,
                *center_x,
                *center_y,
                *radius_x,
                radius_y.unwrap_or(*radius_x),
                *feather,
            ),
            *invert,
        ),
        MaskShape::LinearGradient {
            start_x,
            start_y,
            end_x,
            end_y,
            feather,
            invert,
        } => apply_invert(
            linear_gradient_mask(
                context.x, context.y, *start_x, *start_y, *end_x, *end_y, *feather,
            ),
            *invert,
        ),
        MaskShape::Rectangle {
            left,
            top,
            right,
            bottom,
            feather,
            invert,
        } => apply_invert(
            rectangle_mask(context.x, context.y, *left, *top, *right, *bottom, *feather),
            *invert,
        ),
        MaskShape::LuminanceRange {
            min,
            max,
            feather,
            invert,
        } => apply_invert(
            luminance_range_mask(context.luma, *min, *max, *feather),
            *invert,
        ),
    }
}

fn apply_invert(weight: f32, invert: bool) -> f32 {
    let weight = weight.clamp(0.0, 1.0);
    if invert { 1.0 - weight } else { weight }
}

fn radial_mask(
    x: f32,
    y: f32,
    center_x: f32,
    center_y: f32,
    radius_x: f32,
    radius_y: f32,
    feather: f32,
) -> f32 {
    let radius_x = radius_x.abs().max(0.001);
    let radius_y = radius_y.abs().max(0.001);
    let distance =
        (((x - center_x) / radius_x).powi(2) + ((y - center_y) / radius_y).powi(2)).sqrt();
    feathered_distance(distance, feather)
}

fn rectangle_mask(
    x: f32,
    y: f32,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    feather: f32,
) -> f32 {
    let feather = feather.clamp(0.0, 0.99);
    let left = left.min(right);
    let right = right.max(left);
    let top = top.min(bottom);
    let bottom = bottom.max(top);
    let x_weight =
        smoothstep(left - feather, left, x) * (1.0 - smoothstep(right, right + feather, x));
    let y_weight =
        smoothstep(top - feather, top, y) * (1.0 - smoothstep(bottom, bottom + feather, y));
    (x_weight * y_weight).clamp(0.0, 1.0)
}

fn linear_gradient_mask(
    x: f32,
    y: f32,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    feather: f32,
) -> f32 {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let length_sq = dx * dx + dy * dy;
    if length_sq <= f32::EPSILON {
        return 1.0;
    }

    let t = ((x - start_x) * dx + (y - start_y) * dy) / length_sq;
    let feather = feather.clamp(0.0, 1.0);
    smoothstep(0.0 - feather, 1.0 + feather, t)
}

fn luminance_range_mask(luma: f32, min: f32, max: f32, feather: f32) -> f32 {
    let feather = feather.clamp(0.0, 1.0);
    let min = min.clamp(0.0, 1.0);
    let max = max.clamp(min, 1.0);
    let enter = smoothstep(min - feather, min, luma);
    let exit = 1.0 - smoothstep(max, max + feather, luma);
    (enter * exit).clamp(0.0, 1.0)
}

fn feathered_distance(distance: f32, feather: f32) -> f32 {
    let feather = feather.clamp(0.0, 0.99);
    let inner = (1.0 - feather).clamp(0.0, 1.0);
    1.0 - smoothstep(inner, 1.0, distance)
}

fn blend_channel(base: u8, edited: u8, weight: f32) -> u8 {
    let weight = weight.clamp(0.0, 1.0);
    let blended = base as f32 + (edited as f32 - base as f32) * weight;
    blended.round().clamp(0.0, 255.0) as u8
}

fn prepare_curve_points(curve: &ToneCurve) -> Vec<[f32; 2]> {
    let mut points: Vec<[f32; 2]> = curve
        .points
        .iter()
        .map(|point| [point[0].clamp(0.0, 1.0), point[1].clamp(0.0, 1.0)])
        .collect();
    points.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    if points.is_empty() {
        return ToneCurve::linear().points;
    }
    if points[0][0] > 0.0 {
        points.insert(0, [0.0, 0.0]);
    }
    if points.last().is_some_and(|point| point[0] < 1.0) {
        points.push([1.0, 1.0]);
    }
    points
}

fn evaluate_curve(points: &[[f32; 2]], x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);

    for pair in points.windows(2) {
        let start = pair[0];
        let end = pair[1];
        if x >= start[0] && x <= end[0] {
            let width = (end[0] - start[0]).max(f32::EPSILON);
            let t = (x - start[0]) / width;
            return start[1] + (end[1] - start[1]) * t;
        }
    }

    points.last().map(|point| point[1]).unwrap_or(x)
}

fn hsl_color_center(color: HslColor) -> f32 {
    match color {
        HslColor::Red => 0.0,
        HslColor::Orange => 30.0,
        HslColor::Yellow => 60.0,
        HslColor::Green => 120.0,
        HslColor::Aqua => 180.0,
        HslColor::Blue => 240.0,
        HslColor::Purple => 275.0,
        HslColor::Magenta => 320.0,
    }
}

fn hue_weight(hue: f32, center: f32, width: f32) -> f32 {
    let distance = hue_distance(hue, center);
    (1.0 - distance / width).clamp(0.0, 1.0)
}

fn hue_distance(a: f32, b: f32) -> f32 {
    let delta = (a - b).rem_euclid(360.0);
    delta.min(360.0 - delta)
}

fn rgb_luma(r: f32, g: f32, b: f32) -> f32 {
    (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let width = (edge1 - edge0).max(f32::EPSILON);
    let t = ((x - edge0) / width).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
    use photo_core::{
        BasicAdjustments, HslColor, LocalAdjustmentLayer, MaskShape, PhotoEditRecipe, ToneCurve,
    };

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
        let tensor =
            image_to_nchw_f32_with_normalization(&input, TensorNormalization::ZeroTwoFiftyFive);
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

    #[test]
    fn photo_edit_recipe_stage_keeps_shape() {
        let input = ImageFrame::new(2, 1, vec![10, 20, 30, 200, 100, 50]).expect("valid frame");
        let out = apply_photo_edit_recipe(input.clone(), &PhotoEditRecipe::default())
            .expect("photo edit ok");
        assert_eq!(out.width, input.width);
        assert_eq!(out.height, input.height);
        assert_eq!(out.data.len(), input.data.len());
    }

    #[test]
    fn tone_curve_can_brighten_midtones() {
        let input = ImageFrame::new(1, 1, vec![64, 64, 64]).expect("valid frame");
        let out = apply_tone_curve(
            input,
            &ToneCurve {
                points: vec![[0.0, 0.0], [0.25, 0.45], [1.0, 1.0]],
            },
        )
        .expect("curve ok");
        assert!(out.data[0] > 64);
        assert_eq!(out.data[0], out.data[1]);
        assert_eq!(out.data[1], out.data[2]);
    }

    #[test]
    fn hsl_adjustments_target_matching_hues() {
        let input = ImageFrame::new(1, 1, vec![40, 80, 220]).expect("valid frame");
        let mut recipe = PhotoEditRecipe::default();
        recipe.hsl.insert(
            HslColor::Blue,
            HslAdjustment {
                hue: 0.0,
                saturation: 0.4,
                luminance: -0.2,
            },
        );

        let out = apply_photo_edit_recipe(input.clone(), &recipe).expect("photo edit ok");
        assert_ne!(out.data, input.data);
        assert!(out.data[2] >= out.data[0]);
    }

    #[test]
    fn shadows_lift_dark_pixels_more_than_bright_pixels() {
        let input = ImageFrame::new(2, 1, vec![24, 24, 24, 220, 220, 220]).expect("valid frame");
        let out = apply_global_adjustments(
            input.clone(),
            GlobalAdjustments {
                shadows: 0.7,
                ..GlobalAdjustments::default()
            },
        )
        .expect("global adjustments ok");

        let dark_delta = out.data[0] as i32 - input.data[0] as i32;
        let bright_delta = out.data[3] as i32 - input.data[3] as i32;
        assert!(dark_delta > bright_delta);
    }

    #[test]
    fn highlights_reduce_bright_pixels_more_than_shadow_pixels() {
        let input = ImageFrame::new(2, 1, vec![24, 24, 24, 240, 240, 240]).expect("valid frame");
        let out = apply_global_adjustments(
            input.clone(),
            GlobalAdjustments {
                highlights: -0.8,
                ..GlobalAdjustments::default()
            },
        )
        .expect("global adjustments ok");

        let dark_delta = input.data[0] as i32 - out.data[0] as i32;
        let bright_delta = input.data[3] as i32 - out.data[3] as i32;
        assert!(bright_delta > dark_delta);
    }

    #[test]
    fn hue_shift_wraps_full_turns() {
        let input = ImageFrame::new(1, 1, vec![10, 200, 50]).expect("valid frame");
        let neutral = apply_global_adjustments(input.clone(), GlobalAdjustments::default())
            .expect("global adjustments ok");
        let wrapped = apply_global_adjustments(
            input,
            GlobalAdjustments {
                hue_shift_degrees: 720.0,
                ..GlobalAdjustments::default()
            },
        )
        .expect("global adjustments ok");

        assert_eq!(neutral.data, wrapped.data);
    }

    #[test]
    fn vibrance_favors_desaturated_colors() {
        let low_sat = apply_vibrance(0.15, 0.6, 210.0, 0.6);
        let high_sat = apply_vibrance(0.85, 0.6, 210.0, 0.6);
        assert!(low_sat - 0.15 > high_sat - 0.85);
    }

    #[test]
    fn local_radial_mask_can_isolate_center_pixels() {
        let input = ImageFrame::new(3, 1, vec![100, 100, 100, 100, 100, 100, 100, 100, 100])
            .expect("valid frame");
        let layer = LocalAdjustmentLayer {
            name: Some("center lift".into()),
            opacity: 1.0,
            mask: MaskShape::Radial {
                center_x: 0.5,
                center_y: 0.5,
                radius_x: 0.2,
                radius_y: Some(0.8),
                feather: 0.2,
                invert: false,
            },
            global: GlobalAdjustments {
                exposure: 0.5,
                ..GlobalAdjustments::default()
            },
            tone_curve: None,
            hsl: BTreeMap::new(),
        };

        let out = apply_local_adjustment_layer(input.clone(), &layer).expect("local layer ok");
        assert!(out.data[3] > input.data[3]);
        assert_eq!(out.data[0], input.data[0]);
        assert_eq!(out.data[6], input.data[6]);
    }

    #[test]
    fn luminance_range_mask_targets_highlights() {
        let input = ImageFrame::new(2, 1, vec![32, 32, 32, 220, 220, 220]).expect("valid frame");
        let layer = LocalAdjustmentLayer {
            name: Some("recover highlights".into()),
            opacity: 1.0,
            mask: MaskShape::LuminanceRange {
                min: 0.7,
                max: 1.0,
                feather: 0.1,
                invert: false,
            },
            global: GlobalAdjustments {
                highlights: -0.7,
                ..GlobalAdjustments::default()
            },
            tone_curve: None,
            hsl: BTreeMap::new(),
        };

        let out = apply_local_adjustment_layer(input.clone(), &layer).expect("local layer ok");
        assert_eq!(out.data[0], input.data[0]);
        assert!(out.data[3] < input.data[3]);
    }
}
