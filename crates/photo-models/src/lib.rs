//! 照片处理模型封装与辅助逻辑。

mod face_beauty;
/// 风格迁移模型封装。
mod style_transfer;

pub use face_beauty::{
    DetectedFace, FaceBeautyProcessor, FaceBoundingBox, FaceDetectorModel, FaceLandmarkModel,
};
/// 风格迁移模型配置与运行器。
pub use style_transfer::{
    StyleTransferModel, StyleTransferNormalization, StyleTransferResizePolicy,
};
