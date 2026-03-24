use std::path::{Path, PathBuf};

use photo_core::{BeautySettings, ImageFrame, PhotoError, Result};
use photo_imageops::{
    CurveWarpPoint, EyeWarpPoint, FaceWarpAnchors, TensorNormalization, apply_beauty_filters,
    apply_face_reshape, image_to_nchw_f32_with_normalization, resize_rgb,
};
use photo_onnx::{NamedTensor, OnnxEngine, SessionOptions};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBoundingBox {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub score: f32,
}

impl FaceBoundingBox {
    pub fn width(self) -> f32 {
        (self.right - self.left).max(0.0)
    }

    pub fn height(self) -> f32 {
        (self.bottom - self.top).max(0.0)
    }

    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(self) -> [f32; 2] {
        [
            (self.left + self.right) * 0.5,
            (self.top + self.bottom) * 0.5,
        ]
    }

    pub fn clamped(self) -> Self {
        let left = self.left.clamp(0.0, 1.0);
        let top = self.top.clamp(0.0, 1.0);
        let right = self.right.clamp(left, 1.0);
        let bottom = self.bottom.clamp(top, 1.0);
        Self {
            left,
            top,
            right,
            bottom,
            score: self.score,
        }
    }

    fn expand(self, padding: f32) -> Self {
        let padding_x = self.width() * padding;
        let padding_y = self.height() * padding;
        Self {
            left: self.left - padding_x,
            top: self.top - padding_y,
            right: self.right + padding_x,
            bottom: self.bottom + padding_y,
            score: self.score,
        }
        .clamped()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedFace {
    pub bbox: FaceBoundingBox,
}

#[derive(Debug, Clone)]
pub struct FaceDetectorModel {
    pub model_path: PathBuf,
    pub input_name: String,
    pub boxes_output_name: Option<String>,
    pub scores_output_name: Option<String>,
    pub input_size: (u32, u32),
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub normalization: TensorNormalization,
}

impl FaceDetectorModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            input_name: "input".into(),
            boxes_output_name: None,
            scores_output_name: None,
            input_size: (320, 240),
            score_threshold: 0.65,
            nms_threshold: 0.30,
            normalization: TensorNormalization::MinusOneOne,
        }
    }

    pub fn detect<E: OnnxEngine>(
        &self,
        engine: &mut E,
        image: &ImageFrame,
    ) -> Result<Vec<DetectedFace>> {
        let resized = resize_rgb(image, self.input_size.0, self.input_size.1)?;
        let outputs = run_model_with_candidates(
            engine,
            &self.model_path,
            &self.input_name_candidates(),
            vec![1, 3, self.input_size.1 as usize, self.input_size.0 as usize],
            image_to_nchw_f32_with_normalization(&resized, self.normalization),
        )?;

        let boxes_tensor = select_output_tensor(&outputs, &self.boxes_output_name, can_be_boxes)?;
        let scores_tensor =
            select_output_tensor(&outputs, &self.scores_output_name, can_be_scores)?;

        let boxes = decode_box_rows(boxes_tensor)?;
        let scores = decode_score_rows(scores_tensor)?;
        let mut detections = Vec::new();
        for (bbox, score) in boxes.into_iter().zip(scores) {
            if score < self.score_threshold {
                continue;
            }
            let bbox = FaceBoundingBox {
                left: bbox[0].min(bbox[2]),
                top: bbox[1].min(bbox[3]),
                right: bbox[0].max(bbox[2]),
                bottom: bbox[1].max(bbox[3]),
                score,
            }
            .clamped();

            if bbox.area() > 0.0 {
                detections.push(DetectedFace { bbox });
            }
        }

        detections.sort_by(|a, b| b.bbox.score.total_cmp(&a.bbox.score));
        Ok(non_max_suppression(detections, self.nms_threshold))
    }

    fn input_name_candidates(&self) -> Vec<String> {
        collect_candidates(&self.input_name, &["input", "input0", "data", "image", "x"])
    }
}

#[derive(Debug, Clone)]
pub struct FaceLandmarkModel {
    pub model_path: PathBuf,
    pub input_name: String,
    pub output_name: Option<String>,
    pub input_size: (u32, u32),
    pub normalization: TensorNormalization,
    pub crop_padding: f32,
}

impl FaceLandmarkModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            input_name: "input".into(),
            output_name: None,
            input_size: (112, 112),
            normalization: TensorNormalization::ZeroOne,
            crop_padding: 0.18,
        }
    }

    pub fn predict<E: OnnxEngine>(
        &self,
        engine: &mut E,
        image: &ImageFrame,
        bbox: FaceBoundingBox,
    ) -> Result<Option<Vec<[f32; 2]>>> {
        let crop_rect = bbox.expand(self.crop_padding);
        let cropped = crop_and_resize(image, crop_rect, self.input_size)?;
        let outputs = run_model_with_candidates(
            engine,
            &self.model_path,
            &self.input_name_candidates(),
            vec![1, 3, self.input_size.1 as usize, self.input_size.0 as usize],
            image_to_nchw_f32_with_normalization(&cropped, self.normalization),
        )?;

        let output = select_output_tensor(&outputs, &self.output_name, |tensor| {
            tensor.data.len() >= 2 && tensor.data.len() % 2 == 0
        })?;
        let landmark_rows = decode_landmark_rows(output)?;
        if landmark_rows.is_empty() {
            return Ok(None);
        }

        Ok(Some(
            landmark_rows
                .into_iter()
                .map(|[x, y]| {
                    [
                        crop_rect.left + crop_rect.width() * normalize_coordinate(x),
                        crop_rect.top + crop_rect.height() * normalize_coordinate(y),
                    ]
                })
                .collect(),
        ))
    }

    fn input_name_candidates(&self) -> Vec<String> {
        collect_candidates(&self.input_name, &["input", "input0", "data", "image", "x"])
    }
}

#[derive(Debug, Clone)]
pub struct FaceBeautyProcessor {
    pub detector: FaceDetectorModel,
    pub landmark: FaceLandmarkModel,
}

impl FaceBeautyProcessor {
    pub fn new(detector: FaceDetectorModel, landmark: FaceLandmarkModel) -> Self {
        Self { detector, landmark }
    }

    pub fn process<DE, LE>(
        &self,
        detector_engine: &mut DE,
        landmark_engine: &mut LE,
        image: &ImageFrame,
        settings: BeautySettings,
    ) -> Result<ImageFrame>
    where
        DE: OnnxEngine,
        LE: OnnxEngine,
    {
        let settings = settings.clamped();
        let beauty_applied = apply_beauty_filters(image.clone(), settings)?;
        if !settings.needs_face_detection() {
            return Ok(beauty_applied);
        }

        let faces = self.detector.detect(detector_engine, image)?;
        let Some(primary_face) = select_primary_face(&faces) else {
            return Ok(beauty_applied);
        };

        let Some(landmarks) = self
            .landmark
            .predict(landmark_engine, image, primary_face.bbox)?
        else {
            return Ok(beauty_applied);
        };

        let Some(anchors) = build_face_warp_anchors(&landmarks, primary_face.bbox) else {
            return Ok(beauty_applied);
        };

        apply_face_reshape(
            beauty_applied,
            &anchors,
            settings.thin_face,
            settings.big_eye,
        )
    }
}

pub fn select_primary_face(faces: &[DetectedFace]) -> Option<&DetectedFace> {
    faces.iter().max_by(|a, b| {
        let score_a = primary_face_score(a.bbox);
        let score_b = primary_face_score(b.bbox);
        score_a
            .total_cmp(&score_b)
            .then_with(|| a.bbox.score.total_cmp(&b.bbox.score))
    })
}

fn primary_face_score(bbox: FaceBoundingBox) -> f32 {
    let area_score = bbox.area();
    let center = bbox.center();
    let dx = center[0] - 0.5;
    let dy = center[1] - 0.5;
    let center_distance = (dx * dx + dy * dy).sqrt() / 0.70710677;
    let center_score = (1.0 - center_distance).clamp(0.0, 1.0);
    area_score * 0.7 + center_score * 0.3
}

fn build_face_warp_anchors(
    landmarks: &[[f32; 2]],
    bbox: FaceBoundingBox,
) -> Option<FaceWarpAnchors> {
    if landmarks.len() >= 98 {
        return build_face_warp_anchors_98(landmarks, bbox);
    }
    if landmarks.len() >= 68 {
        return build_face_warp_anchors_68(landmarks, bbox);
    }
    if landmarks.len() < 48 {
        return None;
    }
    build_face_warp_anchors_fallback(landmarks, bbox)
}

fn build_face_warp_anchors_98(
    landmarks: &[[f32; 2]],
    bbox: FaceBoundingBox,
) -> Option<FaceWarpAnchors> {
    let center_x = bbox.center()[0];
    let mut slim_pairs = Vec::new();
    for index in [3usize, 5, 7, 9, 23, 25, 27, 29, 13, 19] {
        let origin = *landmarks.get(index)?;
        slim_pairs.push(CurveWarpPoint {
            origin,
            target: [center_x, origin[1]],
        });
    }

    let left_eye_points = &landmarks[60..68];
    let right_eye_points = &landmarks[68..76];
    let left_eye_center = landmarks
        .get(96)
        .copied()
        .unwrap_or_else(|| centroid(left_eye_points));
    let right_eye_center = landmarks
        .get(97)
        .copied()
        .unwrap_or_else(|| centroid(right_eye_points));

    let left_eye_radius = eye_radius(left_eye_points, bbox);
    let right_eye_radius = eye_radius(right_eye_points, bbox);

    Some(FaceWarpAnchors {
        slim_pairs,
        eyes: vec![
            EyeWarpPoint {
                center: left_eye_center,
                radius: left_eye_radius,
            },
            EyeWarpPoint {
                center: right_eye_center,
                radius: right_eye_radius,
            },
        ],
    })
}

fn build_face_warp_anchors_68(
    landmarks: &[[f32; 2]],
    bbox: FaceBoundingBox,
) -> Option<FaceWarpAnchors> {
    let center_x = bbox.center()[0];
    let mut slim_pairs = Vec::new();
    for index in [2usize, 4, 6, 8, 10, 12, 14] {
        let origin = *landmarks.get(index)?;
        slim_pairs.push(CurveWarpPoint {
            origin,
            target: [center_x, origin[1]],
        });
    }

    let left_eye_points = &landmarks[36..42];
    let right_eye_points = &landmarks[42..48];
    let left_eye_center = centroid(left_eye_points);
    let right_eye_center = centroid(right_eye_points);

    let left_eye_radius = eye_radius(left_eye_points, bbox);
    let right_eye_radius = eye_radius(right_eye_points, bbox);

    Some(FaceWarpAnchors {
        slim_pairs,
        eyes: vec![
            EyeWarpPoint {
                center: left_eye_center,
                radius: left_eye_radius,
            },
            EyeWarpPoint {
                center: right_eye_center,
                radius: right_eye_radius,
            },
        ],
    })
}

fn build_face_warp_anchors_fallback(
    landmarks: &[[f32; 2]],
    bbox: FaceBoundingBox,
) -> Option<FaceWarpAnchors> {
    let center_x = bbox.center()[0];
    let mut slim_pairs = Vec::new();
    let cheek_band = bbox.top + bbox.height() * 0.42;
    let jaw_band = bbox.top + bbox.height() * 0.68;
    let left_edge = bbox.left + bbox.width() * 0.14;
    let right_edge = bbox.right - bbox.width() * 0.14;

    for origin in [
        [left_edge, cheek_band],
        [left_edge, jaw_band],
        [right_edge, cheek_band],
        [right_edge, jaw_band],
    ] {
        slim_pairs.push(CurveWarpPoint {
            origin,
            target: [center_x, origin[1]],
        });
    }

    let eye_slice = if landmarks.len() >= 48 {
        Some((&landmarks[36..42], &landmarks[42..48]))
    } else {
        None
    };

    let (left_eye_center, right_eye_center) = if let Some((left, right)) = eye_slice {
        (centroid(left), centroid(right))
    } else {
        (
            [
                bbox.left + bbox.width() * 0.34,
                bbox.top + bbox.height() * 0.38,
            ],
            [
                bbox.right - bbox.width() * 0.34,
                bbox.top + bbox.height() * 0.38,
            ],
        )
    };

    Some(FaceWarpAnchors {
        slim_pairs,
        eyes: vec![
            EyeWarpPoint {
                center: left_eye_center,
                radius: bbox.width() * 0.22,
            },
            EyeWarpPoint {
                center: right_eye_center,
                radius: bbox.width() * 0.22,
            },
        ],
    })
}

fn eye_radius(points: &[[f32; 2]], bbox: FaceBoundingBox) -> f32 {
    let min_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min);
    let max_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min);
    let max_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let width = (max_x - min_x).max(0.001);
    let height = (max_y - min_y).max(0.001);
    (width.max(height * 1.3) * 2.1).max(bbox.width() * 0.07)
}

fn centroid(points: &[[f32; 2]]) -> [f32; 2] {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for point in points {
        sum_x += point[0];
        sum_y += point[1];
    }
    [sum_x / points.len() as f32, sum_y / points.len() as f32]
}

fn crop_and_resize(
    image: &ImageFrame,
    rect: FaceBoundingBox,
    size: (u32, u32),
) -> Result<ImageFrame> {
    let left = (rect.left * image.width as f32).floor() as u32;
    let top = (rect.top * image.height as f32).floor() as u32;
    let right = (rect.right * image.width as f32).ceil() as u32;
    let bottom = (rect.bottom * image.height as f32).ceil() as u32;
    let crop_width = right.saturating_sub(left).max(1);
    let crop_height = bottom.saturating_sub(top).max(1);

    let mut data = vec![0u8; crop_width as usize * crop_height as usize * 3];
    for y in 0..crop_height as usize {
        let source_y = (top as usize + y).min(image.height.saturating_sub(1) as usize);
        for x in 0..crop_width as usize {
            let source_x = (left as usize + x).min(image.width.saturating_sub(1) as usize);
            let src_idx = (source_y * image.width as usize + source_x) * 3;
            let dst_idx = (y * crop_width as usize + x) * 3;
            data[dst_idx..dst_idx + 3].copy_from_slice(&image.data[src_idx..src_idx + 3]);
        }
    }

    let cropped = ImageFrame::new(crop_width, crop_height, data)?;
    resize_rgb(&cropped, size.0, size.1)
}

fn run_model_with_candidates<E: OnnxEngine>(
    engine: &mut E,
    model_path: &Path,
    input_names: &[String],
    input_shape: Vec<usize>,
    input_data: Vec<f32>,
) -> Result<Vec<NamedTensor>> {
    engine
        .load_model(model_path, SessionOptions::default())
        .map_err(|error| PhotoError::Model(error.to_string()))?;

    let mut last_error = None;
    for input_name in input_names {
        let input = NamedTensor {
            name: input_name.clone(),
            shape: input_shape.clone(),
            data: input_data.clone(),
        };
        match engine.run(&[input]) {
            Ok(outputs) => return Ok(outputs),
            Err(error) => last_error = Some(error),
        }
    }

    Err(PhotoError::Model(format!(
        "failed to run model `{}` with candidate inputs {:?}: {}",
        model_path.display(),
        input_names,
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown inference error".into())
    )))
}

fn collect_candidates(primary: &str, fallbacks: &[&str]) -> Vec<String> {
    let mut candidates = vec![primary.to_string()];
    for fallback in fallbacks {
        if !candidates.iter().any(|candidate| candidate == fallback) {
            candidates.push((*fallback).to_string());
        }
    }
    candidates
}

fn select_output_tensor<'a, F>(
    outputs: &'a [NamedTensor],
    explicit_name: &Option<String>,
    predicate: F,
) -> Result<&'a NamedTensor>
where
    F: Fn(&NamedTensor) -> bool,
{
    if let Some(name) = explicit_name {
        return outputs
            .iter()
            .find(|output| &output.name == name)
            .ok_or_else(|| PhotoError::Model(format!("output tensor `{name}` not found")));
    }

    outputs
        .iter()
        .find(|output| predicate(output))
        .ok_or_else(|| PhotoError::Model("unable to infer model output tensor".into()))
}

fn can_be_boxes(tensor: &NamedTensor) -> bool {
    last_dim(tensor.shape.as_slice()) == Some(4)
        || (tensor.shape.len() == 3 && tensor.shape[0] == 1 && tensor.shape[1] == 4)
}

fn can_be_scores(tensor: &NamedTensor) -> bool {
    matches!(last_dim(tensor.shape.as_slice()), Some(1 | 2))
        || (tensor.shape.len() == 3
            && tensor.shape[0] == 1
            && matches!(tensor.shape[1], 1 | 2)
            && last_dim(tensor.shape.as_slice()) != Some(4))
}

fn decode_box_rows(tensor: &NamedTensor) -> Result<Vec<[f32; 4]>> {
    if let Some(rows) = decode_rows_last_dim(tensor, 4) {
        return Ok(rows
            .into_iter()
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect());
    }
    if let Some(rows) = decode_rows_channel_plane(tensor, 4) {
        return Ok(rows
            .into_iter()
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect());
    }
    Err(PhotoError::Model(format!(
        "unsupported face detector box tensor shape {:?}",
        tensor.shape
    )))
}

fn decode_score_rows(tensor: &NamedTensor) -> Result<Vec<f32>> {
    if let Some(rows) = decode_rows_last_dim(tensor, 2) {
        return Ok(rows.into_iter().map(|row| row[1]).collect());
    }
    if let Some(rows) = decode_rows_channel_plane(tensor, 2) {
        return Ok(rows.into_iter().map(|row| row[1]).collect());
    }
    if let Some(rows) = decode_rows_last_dim(tensor, 1) {
        return Ok(rows.into_iter().map(|row| row[0]).collect());
    }
    Err(PhotoError::Model(format!(
        "unsupported face detector score tensor shape {:?}",
        tensor.shape
    )))
}

fn decode_landmark_rows(tensor: &NamedTensor) -> Result<Vec<[f32; 2]>> {
    if let Some(rows) = decode_rows_last_dim(tensor, 2) {
        return Ok(rows.into_iter().map(|row| [row[0], row[1]]).collect());
    }
    if let Some(rows) = decode_rows_channel_plane(tensor, 2) {
        return Ok(rows.into_iter().map(|row| [row[0], row[1]]).collect());
    }
    if tensor.data.len() % 2 == 0 {
        return Ok(tensor
            .data
            .chunks_exact(2)
            .map(|row| [row[0], row[1]])
            .collect());
    }

    Err(PhotoError::Model(format!(
        "unsupported landmark tensor shape {:?}",
        tensor.shape
    )))
}

fn decode_rows_last_dim(tensor: &NamedTensor, columns: usize) -> Option<Vec<Vec<f32>>> {
    if last_dim(tensor.shape.as_slice()) != Some(columns) {
        return None;
    }
    Some(
        tensor
            .data
            .chunks_exact(columns)
            .map(|chunk| chunk.to_vec())
            .collect(),
    )
}

fn decode_rows_channel_plane(tensor: &NamedTensor, channels: usize) -> Option<Vec<Vec<f32>>> {
    if tensor.shape.len() != 3 || tensor.shape[0] != 1 || tensor.shape[1] != channels {
        return None;
    }
    let rows = tensor.shape[2];
    let row_width = tensor.shape[2];
    let mut decoded = vec![vec![0.0; channels]; rows];
    for channel in 0..channels {
        let channel_offset = channel * row_width;
        for row in 0..rows {
            decoded[row][channel] = tensor.data[channel_offset + row];
        }
    }
    Some(decoded)
}

fn last_dim(shape: &[usize]) -> Option<usize> {
    shape.last().copied()
}

fn normalize_coordinate(value: f32) -> f32 {
    if (-1.0..=1.0).contains(&value) && !(0.0..=1.0).contains(&value) {
        ((value + 1.0) * 0.5).clamp(0.0, 1.0)
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn non_max_suppression(mut detections: Vec<DetectedFace>, threshold: f32) -> Vec<DetectedFace> {
    let mut kept = Vec::new();
    while let Some(candidate) = detections.first().cloned() {
        detections.remove(0);
        detections.retain(|detection| iou(candidate.bbox, detection.bbox) < threshold);
        kept.push(candidate);
    }
    kept
}

fn iou(a: FaceBoundingBox, b: FaceBoundingBox) -> f32 {
    let left = a.left.max(b.left);
    let top = a.top.max(b.top);
    let right = a.right.min(b.right);
    let bottom = a.bottom.min(b.bottom);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union = a.area() + b.area() - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use photo_onnx::{OnnxError, Result as OnnxResult};

    use super::*;

    #[derive(Default)]
    struct MockEngine {
        accepted_input_name: Option<String>,
        outputs: Vec<NamedTensor>,
        seen_input_shape: Option<Vec<usize>>,
        load_calls: usize,
    }

    impl OnnxEngine for MockEngine {
        fn backend_name(&self) -> &'static str {
            "mock"
        }

        fn load_model<P: AsRef<Path>>(
            &mut self,
            _path: P,
            _options: SessionOptions,
        ) -> OnnxResult<()> {
            self.load_calls += 1;
            Ok(())
        }

        fn run(&mut self, inputs: &[NamedTensor]) -> OnnxResult<Vec<NamedTensor>> {
            let input = inputs
                .first()
                .ok_or_else(|| OnnxError::InvalidTensor("missing input".into()))?;
            self.seen_input_shape = Some(input.shape.clone());
            if let Some(name) = &self.accepted_input_name {
                if &input.name != name {
                    return Err(OnnxError::Inference(format!(
                        "unexpected input {}",
                        input.name
                    )));
                }
            }
            Ok(self.outputs.clone())
        }
    }

    fn build_landmark_output() -> Vec<f32> {
        let mut points = Vec::with_capacity(98 * 2);
        for index in 0..98 {
            let x = if index < 33 {
                0.1 + index as f32 * 0.025
            } else if (60..68).contains(&index) {
                0.32 + (index - 60) as f32 * 0.01
            } else if (68..76).contains(&index) {
                0.58 + (index - 68) as f32 * 0.01
            } else if index == 96 {
                0.36
            } else if index == 97 {
                0.62
            } else {
                0.5
            };
            let y = if (60..76).contains(&index) { 0.42 } else { 0.6 };
            points.push(x);
            points.push(y);
        }
        points
    }

    #[test]
    fn detector_decodes_boxes_and_scores() {
        let model = FaceDetectorModel::new("detector.onnx");
        let image = ImageFrame::new(4, 4, vec![0; 4 * 4 * 3]).expect("frame");
        let mut engine = MockEngine {
            accepted_input_name: Some("input".into()),
            outputs: vec![
                NamedTensor {
                    name: "boxes".into(),
                    shape: vec![1, 3, 4],
                    data: vec![
                        0.1, 0.1, 0.4, 0.4, //
                        0.2, 0.2, 0.8, 0.8, //
                        0.7, 0.1, 0.9, 0.3,
                    ],
                },
                NamedTensor {
                    name: "scores".into(),
                    shape: vec![1, 3, 2],
                    data: vec![
                        0.1, 0.82, //
                        0.1, 0.91, //
                        0.6, 0.62,
                    ],
                },
            ],
            ..MockEngine::default()
        };

        let detections = model.detect(&mut engine, &image).expect("detect");
        assert_eq!(detections.len(), 2);
        assert_eq!(engine.seen_input_shape, Some(vec![1, 3, 240, 320]));
        assert!(detections[0].bbox.score >= detections[1].bbox.score);
    }

    #[test]
    fn primary_face_prefers_large_centered_face() {
        let faces = vec![
            DetectedFace {
                bbox: FaceBoundingBox {
                    left: 0.0,
                    top: 0.0,
                    right: 0.2,
                    bottom: 0.2,
                    score: 0.98,
                },
            },
            DetectedFace {
                bbox: FaceBoundingBox {
                    left: 0.2,
                    top: 0.2,
                    right: 0.75,
                    bottom: 0.85,
                    score: 0.88,
                },
            },
        ];

        let primary = select_primary_face(&faces).expect("primary face");
        assert_eq!(primary.bbox.left, 0.2);
    }

    #[test]
    fn landmark_model_maps_points_back_into_image_space() {
        let model = FaceLandmarkModel::new("landmark.onnx");
        let image = ImageFrame::new(200, 100, vec![128; 200 * 100 * 3]).expect("frame");
        let bbox = FaceBoundingBox {
            left: 0.25,
            top: 0.2,
            right: 0.75,
            bottom: 0.9,
            score: 0.9,
        };
        let mut engine = MockEngine {
            outputs: vec![NamedTensor {
                name: "landmarks".into(),
                shape: vec![1, 196],
                data: build_landmark_output(),
            }],
            ..MockEngine::default()
        };

        let landmarks = model
            .predict(&mut engine, &image, bbox)
            .expect("predict")
            .expect("landmarks");
        let crop_rect = bbox.expand(model.crop_padding);
        assert_eq!(landmarks.len(), 98);
        assert!(landmarks[0][0] >= crop_rect.left);
        assert!(landmarks[0][0] <= crop_rect.right);
    }

    #[test]
    fn processor_skips_geometry_when_no_face_is_detected() {
        let processor = FaceBeautyProcessor::new(
            FaceDetectorModel::new("detector.onnx"),
            FaceLandmarkModel::new("landmark.onnx"),
        );
        let input = ImageFrame::new(2, 1, vec![20, 30, 40, 200, 180, 160]).expect("frame");
        let settings = BeautySettings {
            thin_face: 0.6,
            ..BeautySettings::default()
        };
        let mut detector_engine = MockEngine {
            outputs: vec![
                NamedTensor {
                    name: "boxes".into(),
                    shape: vec![1, 1, 4],
                    data: vec![0.1, 0.1, 0.5, 0.5],
                },
                NamedTensor {
                    name: "scores".into(),
                    shape: vec![1, 1, 2],
                    data: vec![0.7, 0.2],
                },
            ],
            ..MockEngine::default()
        };
        let mut landmark_engine = MockEngine::default();

        let output = processor
            .process(&mut detector_engine, &mut landmark_engine, &input, settings)
            .expect("process");
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn processor_runs_beauty_without_model_calls_when_only_skin_filters_are_enabled() {
        let processor = FaceBeautyProcessor::new(
            FaceDetectorModel::new("detector.onnx"),
            FaceLandmarkModel::new("landmark.onnx"),
        );
        let input = ImageFrame::new(2, 1, vec![210, 170, 145, 20, 60, 180]).expect("frame");
        let settings = BeautySettings {
            whiteness: 0.8,
            ..BeautySettings::default()
        };
        let mut detector_engine = MockEngine::default();
        let mut landmark_engine = MockEngine::default();

        let output = processor
            .process(&mut detector_engine, &mut landmark_engine, &input, settings)
            .expect("process");
        assert_ne!(output.data, input.data);
        assert_eq!(detector_engine.load_calls, 0);
        assert_eq!(landmark_engine.load_calls, 0);
    }

    #[test]
    fn warp_anchor_builder_supports_68_point_landmarks() {
        let mut landmarks = vec![[0.5, 0.5]; 68];
        for (idx, point) in landmarks.iter_mut().enumerate().take(17) {
            point[0] = 0.18 + idx as f32 * 0.04;
            point[1] = 0.72 - ((idx as f32 - 8.0).abs() * 0.01);
        }
        for (offset, point) in landmarks[36..42].iter_mut().enumerate() {
            point[0] = 0.34 + offset as f32 * 0.015;
            point[1] = 0.40 + (offset % 2) as f32 * 0.01;
        }
        for (offset, point) in landmarks[42..48].iter_mut().enumerate() {
            point[0] = 0.56 + offset as f32 * 0.015;
            point[1] = 0.40 + (offset % 2) as f32 * 0.01;
        }

        let anchors = build_face_warp_anchors(
            &landmarks,
            FaceBoundingBox {
                left: 0.2,
                top: 0.15,
                right: 0.8,
                bottom: 0.9,
                score: 0.95,
            },
        )
        .expect("anchors");

        assert!(!anchors.slim_pairs.is_empty());
        assert_eq!(anchors.eyes.len(), 2);
        assert!(anchors.eyes[0].radius > 0.0);
    }
}
