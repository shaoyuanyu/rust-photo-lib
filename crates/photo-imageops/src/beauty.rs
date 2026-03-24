use photo_core::{BeautySettings, ImageFrame, Result};

use crate::resize_rgb;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveWarpPoint {
    pub origin: [f32; 2],
    pub target: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeWarpPoint {
    pub center: [f32; 2],
    pub radius: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FaceWarpAnchors {
    pub slim_pairs: Vec<CurveWarpPoint>,
    pub eyes: Vec<EyeWarpPoint>,
}

pub fn apply_beauty_filters(input: ImageFrame, settings: BeautySettings) -> Result<ImageFrame> {
    let settings = settings.clamped();
    let mut frame = input;
    if settings.skin_smoothing > f32::EPSILON || settings.detail_sharpen > f32::EPSILON {
        frame = apply_skin_smoothing(frame, settings.skin_smoothing, settings.detail_sharpen)?;
    }
    if settings.whiteness > f32::EPSILON {
        frame = apply_whiteness(frame, settings.whiteness)?;
    }
    Ok(frame)
}

pub fn apply_skin_smoothing(
    input: ImageFrame,
    skin_smoothing: f32,
    detail_sharpen: f32,
) -> Result<ImageFrame> {
    let skin_smoothing = skin_smoothing.clamp(0.0, 1.0);
    let detail_sharpen = detail_sharpen.clamp(0.0, 1.0);
    if skin_smoothing <= f32::EPSILON && detail_sharpen <= f32::EPSILON {
        return Ok(input);
    }

    let analysis_frame = resize_for_analysis(&input, 1_024)?;
    let original = FloatImage::from_frame(&input);
    let analysis = FloatImage::from_frame(&analysis_frame);
    let mean_small = box_blur_rgb(&analysis, 4);
    let diff_small = compute_local_difference(&analysis, &mean_small, 7.07);
    let mean = resize_float_image(&mean_small, input.width, input.height)?;
    let diff = resize_float_image(&diff_small, input.width, input.height)?;
    let high_pass = high_pass_3x3(&original);

    let theta = 0.1f32;
    let mut out = vec![0.0; original.data.len()];
    for idx in (0..out.len()).step_by(3) {
        let source = [
            original.data[idx],
            original.data[idx + 1],
            original.data[idx + 2],
        ];
        let mean_color = [mean.data[idx], mean.data[idx + 1], mean.data[idx + 2]];
        let variance = [diff.data[idx], diff.data[idx + 1], diff.data[idx + 2]];
        let mean_var = (variance[0] + variance[1] + variance[2]) / 3.0;
        let p = ((source[0].min(mean_color[0] - 0.1) - 0.2) * 4.0).clamp(0.0, 1.0);
        let k_min = ((1.0 - mean_var / (mean_var + theta)) * p * skin_smoothing).clamp(0.0, 1.0);

        for channel in 0..3 {
            let smoothed = mix(source[channel], mean_color[channel], k_min);
            let sharpened = smoothed + detail_sharpen * high_pass.data[idx + channel] * 2.0;
            out[idx + channel] = sharpened.clamp(0.0, 1.0);
        }
    }

    FloatImage {
        width: input.width,
        height: input.height,
        data: out,
    }
    .to_frame()
}

pub fn apply_whiteness(input: ImageFrame, whiteness: f32) -> Result<ImageFrame> {
    let whiteness = whiteness.clamp(0.0, 1.0);
    if whiteness <= f32::EPSILON {
        return Ok(input);
    }

    let mut out = input.data.clone();
    for chunk in out.chunks_exact_mut(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let (h, s, v) = rgb_to_hsv(r, g, b);

        let skin_hue = hue_weight(h, 28.0, 42.0);
        let sat_gate = smoothstep(0.08, 0.22, s) * (1.0 - smoothstep(0.72, 0.95, s));
        let val_gate = smoothstep(0.05, 0.18, v) * (1.0 - smoothstep(0.88, 1.0, v));
        let luma_gate = smoothstep(0.12, 0.32, rgb_luma(r, g, b))
            * (1.0 - smoothstep(0.86, 1.0, rgb_luma(r, g, b)));
        let skin_mask = (skin_hue * sat_gate * val_gate * luma_gate).clamp(0.0, 1.0);
        let lift = whiteness * skin_mask;

        let v2 = (v + 0.18 * lift).clamp(0.0, 1.0);
        let s2 = (s * (1.0 - 0.14 * lift)).clamp(0.0, 1.0);
        let (mut nr, mut ng, mut nb) = hsv_to_rgb(h, s2, v2);

        nr *= 1.0 - 0.010 * lift;
        ng *= 1.0 - 0.018 * lift;
        nb = (nb + 0.028 * lift).clamp(0.0, 1.0);

        chunk[0] = to_u8(nr.clamp(0.0, 1.0));
        chunk[1] = to_u8(ng.clamp(0.0, 1.0));
        chunk[2] = to_u8(nb);
    }

    ImageFrame::new(input.width, input.height, out)
}

pub fn apply_face_reshape(
    input: ImageFrame,
    anchors: &FaceWarpAnchors,
    thin_face: f32,
    big_eye: f32,
) -> Result<ImageFrame> {
    let thin_face = thin_face.clamp(0.0, 1.0);
    let big_eye = big_eye.clamp(0.0, 1.0);
    if thin_face <= f32::EPSILON && big_eye <= f32::EPSILON {
        return Ok(input);
    }
    if anchors.slim_pairs.is_empty() && anchors.eyes.is_empty() {
        return Ok(input);
    }

    let source = FloatImage::from_frame(&input);
    let aspect = input.width as f32 / input.height.max(1) as f32;
    let mut out = vec![0.0; source.data.len()];

    for y in 0..input.height as usize {
        for x in 0..input.width as usize {
            let mut coord = [
                (x as f32 + 0.5) / input.width as f32,
                (y as f32 + 0.5) / input.height as f32,
            ];

            if thin_face > f32::EPSILON {
                for pair in &anchors.slim_pairs {
                    coord = curve_warp(coord, pair.origin, pair.target, thin_face * 0.34, aspect);
                }
            }

            if big_eye > f32::EPSILON {
                for eye in &anchors.eyes {
                    coord = enlarge_eye(coord, eye.center, eye.radius, big_eye * 0.36, aspect);
                }
            }

            let sample = sample_bilinear(&source, coord[0], coord[1]);
            let idx = (y * input.width as usize + x) * 3;
            out[idx] = sample[0];
            out[idx + 1] = sample[1];
            out[idx + 2] = sample[2];
        }
    }

    FloatImage {
        width: input.width,
        height: input.height,
        data: out,
    }
    .to_frame()
}

#[derive(Clone)]
struct FloatImage {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl FloatImage {
    fn from_frame(frame: &ImageFrame) -> Self {
        Self {
            width: frame.width,
            height: frame.height,
            data: frame.data.iter().map(|v| *v as f32 / 255.0).collect(),
        }
    }

    fn to_frame(&self) -> Result<ImageFrame> {
        ImageFrame::new(
            self.width,
            self.height,
            self.data.iter().map(|v| to_u8(v.clamp(0.0, 1.0))).collect(),
        )
    }
}

fn resize_for_analysis(frame: &ImageFrame, max_dim: u32) -> Result<ImageFrame> {
    let longest = frame.width.max(frame.height);
    if longest <= max_dim {
        return Ok(frame.clone());
    }

    let ratio = max_dim as f32 / longest as f32;
    let target_w = ((frame.width as f32 * ratio).round() as u32).max(1);
    let target_h = ((frame.height as f32 * ratio).round() as u32).max(1);
    resize_rgb(frame, target_w, target_h)
}

fn resize_float_image(image: &FloatImage, width: u32, height: u32) -> Result<FloatImage> {
    if image.width == width && image.height == height {
        return Ok(image.clone());
    }
    let frame = image.to_frame()?;
    let resized = resize_rgb(&frame, width, height)?;
    Ok(FloatImage::from_frame(&resized))
}

fn box_blur_rgb(image: &FloatImage, radius: usize) -> FloatImage {
    if radius == 0 || image.width == 0 || image.height == 0 {
        return image.clone();
    }

    let width = image.width as usize;
    let height = image.height as usize;
    let mut horizontal = vec![0.0; image.data.len()];

    for y in 0..height {
        for channel in 0..3 {
            let mut prefix = vec![0.0; width + 1];
            for x in 0..width {
                let idx = (y * width + x) * 3 + channel;
                prefix[x + 1] = prefix[x] + image.data[idx];
            }

            for x in 0..width {
                let left = x.saturating_sub(radius);
                let right = (x + radius).min(width - 1);
                let count = (right - left + 1) as f32;
                let idx = (y * width + x) * 3 + channel;
                horizontal[idx] = (prefix[right + 1] - prefix[left]) / count;
            }
        }
    }

    let mut output = vec![0.0; image.data.len()];
    for x in 0..width {
        for channel in 0..3 {
            let mut prefix = vec![0.0; height + 1];
            for y in 0..height {
                let idx = (y * width + x) * 3 + channel;
                prefix[y + 1] = prefix[y] + horizontal[idx];
            }

            for y in 0..height {
                let top = y.saturating_sub(radius);
                let bottom = (y + radius).min(height - 1);
                let count = (bottom - top + 1) as f32;
                let idx = (y * width + x) * 3 + channel;
                output[idx] = (prefix[bottom + 1] - prefix[top]) / count;
            }
        }
    }

    FloatImage {
        width: image.width,
        height: image.height,
        data: output,
    }
}

fn compute_local_difference(original: &FloatImage, mean: &FloatImage, delta: f32) -> FloatImage {
    let mut output = vec![0.0; original.data.len()];
    for idx in 0..output.len() {
        let diff = (original.data[idx] - mean.data[idx]) * delta;
        output[idx] = (diff * diff).min(1.0);
    }
    FloatImage {
        width: original.width,
        height: original.height,
        data: output,
    }
}

fn high_pass_3x3(image: &FloatImage) -> FloatImage {
    const KERNEL: [[f32; 3]; 3] = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ];

    let width = image.width as usize;
    let height = image.height as usize;
    let mut output = vec![0.0; image.data.len()];

    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let mut blurred = 0.0;
                for ky in 0..3 {
                    let sy = clamp_index(y as isize + ky as isize - 1, height);
                    for kx in 0..3 {
                        let sx = clamp_index(x as isize + kx as isize - 1, width);
                        let idx = (sy * width + sx) * 3 + channel;
                        blurred += image.data[idx] * KERNEL[ky][kx];
                    }
                }
                let idx = (y * width + x) * 3 + channel;
                output[idx] = image.data[idx] - blurred;
            }
        }
    }

    FloatImage {
        width: image.width,
        height: image.height,
        data: output,
    }
}

fn sample_bilinear(image: &FloatImage, u: f32, v: f32) -> [f32; 3] {
    let width = image.width.max(1) as f32;
    let height = image.height.max(1) as f32;
    let x = (u.clamp(0.0, 1.0) * width - 0.5).clamp(0.0, image.width.saturating_sub(1) as f32);
    let y = (v.clamp(0.0, 1.0) * height - 0.5).clamp(0.0, image.height.saturating_sub(1) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(image.width.saturating_sub(1) as usize);
    let y1 = (y0 + 1).min(image.height.saturating_sub(1) as usize);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let top_left = get_pixel(image, x0, y0);
    let top_right = get_pixel(image, x1, y0);
    let bottom_left = get_pixel(image, x0, y1);
    let bottom_right = get_pixel(image, x1, y1);

    let mut result = [0.0; 3];
    for channel in 0..3 {
        let top = mix(top_left[channel], top_right[channel], tx);
        let bottom = mix(bottom_left[channel], bottom_right[channel], tx);
        result[channel] = mix(top, bottom, ty);
    }
    result
}

fn get_pixel(image: &FloatImage, x: usize, y: usize) -> [f32; 3] {
    let idx = (y * image.width as usize + x) * 3;
    [image.data[idx], image.data[idx + 1], image.data[idx + 2]]
}

fn curve_warp(
    texture_coord: [f32; 2],
    origin_position: [f32; 2],
    target_position: [f32; 2],
    delta: f32,
    aspect_ratio: f32,
) -> [f32; 2] {
    let direction = [
        (target_position[0] - origin_position[0]) * delta,
        (target_position[1] - origin_position[1]) * delta,
    ];
    let radius = distance_with_aspect(target_position, origin_position, aspect_ratio);
    if radius <= f32::EPSILON {
        return texture_coord;
    }

    let mut ratio = distance_with_aspect(texture_coord, origin_position, aspect_ratio) / radius;
    ratio = (1.0 - ratio).clamp(0.0, 1.0);

    [
        (texture_coord[0] - direction[0] * ratio).clamp(0.0, 1.0),
        (texture_coord[1] - direction[1] * ratio).clamp(0.0, 1.0),
    ]
}

fn enlarge_eye(
    texture_coord: [f32; 2],
    origin_position: [f32; 2],
    radius: f32,
    delta: f32,
    aspect_ratio: f32,
) -> [f32; 2] {
    if radius <= f32::EPSILON {
        return texture_coord;
    }

    let mut weight = distance_with_aspect(texture_coord, origin_position, aspect_ratio) / radius;
    weight = 1.0 - (1.0 - weight * weight) * delta;
    weight = weight.clamp(0.0, 1.0);

    [
        (origin_position[0] + (texture_coord[0] - origin_position[0]) * weight).clamp(0.0, 1.0),
        (origin_position[1] + (texture_coord[1] - origin_position[1]) * weight).clamp(0.0, 1.0),
    ]
}

fn distance_with_aspect(a: [f32; 2], b: [f32; 2], aspect_ratio: f32) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] / aspect_ratio - b[1] / aspect_ratio;
    (dx * dx + dy * dy).sqrt()
}

fn clamp_index(value: isize, upper_bound: usize) -> usize {
    value.clamp(0, upper_bound.saturating_sub(1) as isize) as usize
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    let hue = if delta <= f32::EPSILON {
        0.0
    } else if (max - r).abs() <= f32::EPSILON {
        60.0 * ((g - b) / delta).rem_euclid(6.0)
    } else if (max - g).abs() <= f32::EPSILON {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let saturation = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    (
        hue.rem_euclid(360.0),
        saturation.clamp(0.0, 1.0),
        max.clamp(0.0, 1.0),
    )
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0);
    let s = s.clamp(0.0, 1.0);
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

fn hue_weight(hue: f32, center: f32, width: f32) -> f32 {
    let distance = ((hue - center + 180.0).rem_euclid(360.0) - 180.0).abs();
    (1.0 - distance / width.max(1.0)).clamp(0.0, 1.0)
}

fn rgb_luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_spreads_impulse_energy() {
        let image = FloatImage {
            width: 3,
            height: 1,
            data: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        };
        let blurred = box_blur_rgb(&image, 1);

        assert!(blurred.data[0] > 0.0);
        assert!(blurred.data[3] < 1.0);
        assert!(blurred.data[6] > 0.0);
    }

    #[test]
    fn local_difference_zero_for_identical_inputs() {
        let image = FloatImage {
            width: 2,
            height: 1,
            data: vec![0.2, 0.3, 0.4, 0.6, 0.5, 0.4],
        };
        let diff = compute_local_difference(&image, &image, 7.07);
        assert!(diff.data.iter().all(|value| *value <= f32::EPSILON));
    }

    #[test]
    fn bilinear_sampling_interpolates_midpoint() {
        let image = FloatImage {
            width: 2,
            height: 2,
            data: vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            ],
        };

        let sample = sample_bilinear(&image, 0.5, 0.5);
        assert!((sample[0] - 0.5).abs() < 0.01);
        assert!((sample[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn beauty_identity_keeps_input() {
        let input = ImageFrame::new(2, 1, vec![32, 64, 96, 128, 160, 192]).expect("frame");
        let output =
            apply_beauty_filters(input.clone(), BeautySettings::default()).expect("beauty");
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn whiteness_prefers_skin_colored_pixels() {
        let input = ImageFrame::new(2, 1, vec![210, 170, 145, 20, 60, 180]).expect("frame");
        let output = apply_whiteness(input.clone(), 1.0).expect("whiteness");

        let skin_delta = output.data[0] as i32 - input.data[0] as i32;
        let blue_delta = output.data[3] as i32 - input.data[3] as i32;
        assert!(skin_delta > blue_delta);
    }

    #[test]
    fn face_reshape_with_anchors_changes_pixels() {
        let input = ImageFrame::new(
            5,
            1,
            vec![10, 0, 0, 60, 0, 0, 120, 0, 0, 180, 0, 0, 240, 0, 0],
        )
        .expect("frame");
        let anchors = FaceWarpAnchors {
            slim_pairs: vec![CurveWarpPoint {
                origin: [0.2, 0.5],
                target: [0.5, 0.5],
            }],
            eyes: Vec::new(),
        };

        let output = apply_face_reshape(input.clone(), &anchors, 1.0, 0.0).expect("reshape");
        assert_ne!(output.data, input.data);
    }
}
