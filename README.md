# rust-photo-lib

Rust 照片处理库（MVP）

当前已实现：
- 基础调色：曝光、亮度、对比度、饱和度、色温、色调（Hue）、Tint
- 可组合 Pipeline API
- ONNX 统一抽象层（`OnnxEngine`）
- 双后端实现：`tract` 与 `ort`（均通过 feature 开关）
- 风格迁移模型适配器（NCHW/f32 预处理与后处理）
- CLI 最小闭环：输入图片 -> 基础调色 -> 可选风格迁移 -> 输出图片

## Workspace 结构

- `crates/photo-core`: 核心数据结构、错误类型、Pipeline/Stage 抽象
- `crates/photo-imageops`: 图像 I/O、基础调色算子、张量转换
- `crates/photo-onnx`: ONNX 后端抽象接口
- `crates/photo-backend-tract`: tract 后端实现
- `crates/photo-backend-ort`: ort 后端实现
- `crates/photo-models`: 模型适配器（当前含 style transfer）
- `crates/photo-cli`: 命令行入口

## 构建

默认（不启用后端）：

```bash
cargo check --workspace
```

启用 tract 后端：

```bash
cargo check -p photo-cli --features tract-backend
```

启用 ort 后端：

```bash
cargo check -p photo-cli --features ort-backend
```

## CLI 使用

仅基础调色：

```bash
cargo run -p photo-cli -- \
  --input input.jpg \
  --output output.jpg \
  --exposure 0.2 \
  --contrast 1.1 \
  --saturation 1.2
```

基础调色 + 风格迁移（tract）：

```bash
cargo run -p photo-cli --features tract-backend -- \
  --input input.jpg \
  --output stylized.jpg \
  --style-model style.onnx \
  --backend tract
```

基础调色 + 风格迁移（ort）：

```bash
cargo run -p photo-cli --features ort-backend -- \
  --input input.jpg \
  --output stylized.jpg \
  --style-model style.onnx
```

显式指定后端（可选）：

```bash
cargo run -p photo-cli --features ort-backend -- \
  --input input.jpg \
  --output stylized.jpg \
  --style-model style.onnx \
  --backend ort
```

## 示例程序

基础调色示例：

```bash
cargo run -p photo-cli --example basic_tone
```

风格迁移示例（需启用 tract）：

```bash
cargo run -p photo-cli --features tract-backend --example style_transfer
```

## 测试与检查

```bash
cargo fmt --all
cargo test --workspace
cargo check -p photo-cli --features tract-backend
cargo check -p photo-cli --features ort-backend
```

## 说明

- 当前模型适配默认输出形状为 `[1, 3, H, W]`。
- 风格迁移默认使用 `[0,255]` 像素值语义（与 `rust-style-transfer` 对齐）。
- 风格迁移默认将输入缩放到 `max_dim=800` 且边长对齐到 `8` 的倍数，再在输出端恢复到原图尺寸。
- 输入节点名默认 `input1`，并内置 `input1/input/image/x` 回退尝试。
- CLI 在启用风格迁移时默认后端为 `ort`。
- `tract` 后端目前对 fast-neural-style 常见 ONNX（如 `*-9.onnx`）中的 `Upsample` 支持有限，可能无法运行；推荐优先使用 `ort`。
