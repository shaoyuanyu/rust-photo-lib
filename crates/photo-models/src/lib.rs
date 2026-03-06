//! 照片处理模型封装与辅助逻辑。

/// 风格迁移模型封装。
mod style_transfer;

/// 风格迁移模型配置与运行器。
pub use style_transfer::{
    StyleTransferModel, StyleTransferNormalization, StyleTransferResizePolicy,
};
