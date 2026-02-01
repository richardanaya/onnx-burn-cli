# 🚀 nnx - Neural Network Executor

<div align="center">

**A blazing-fast GPU-accelerated ONNX inference engine written in Rust** 🦀⚡

[![Rust](https://img.shields.io/badge/Rust-2024-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.0-green.svg)](https://github.com/yourusername/nnx)

</div>

---

## ✨ Features

- 🎮 **GPU Acceleration** - Powered by Burn and wgpu for high-performance inference
- 📦 **ONNX Support** - Run models trained in PyTorch, TensorFlow, and more
- 🔥 **Rust Performance** - Memory-safe and lightning fast
- 🖼️ **Image Processing** - Built-in image loading and preprocessing
- 🏷️ **Easy Inference** - Simple CLI for model inference
- 🔌 **Multi-device** - Support for multiple GPU devices

---

## 🛠️ Installation

### Prerequisites

- 🦀 Rust 2024 edition or later
- 🔧 Cargo (comes with Rust)

### Build from source

```bash
git clone https://github.com/yourusername/nnx.git
cd nnx
cargo build --release
```

---

## 🚀 Getting Started

### List Available Devices

Check what GPUs are available on your system:

```bash
cargo run --release -- devices
```

Output example:
```
🎮 Available GPU Devices:
  [0] NVIDIA GeForce RTX 3080
  [1] AMD Radeon RX 6800 XT
```

### Model Information

Get detailed info about an ONNX model:

```bash
cargo run --release -- info model.onnx
```

### Run Inference

Perform inference on an image with a model:

```bash
cargo run --release -- infer model.onnx \
  --input sample.jpg \
  --labels imagenet_labels.txt \
  --top 5
```

Output example:
```
🔮 Running inference...
✨ Top 5 predictions:
  1. [0.9532] Golden Retriever 🐕
  2. [0.0231] Labrador Retriever 🐕
  3. [0.0124] Cocker Spaniel 🐕
  4. [0.0087] English Setter 🐕
  5. [0.0023] Brittany 🐕
```

---

## 🏗️ Architecture

nnx is built with cutting-edge Rust libraries:

- 📥 **ONNX IR** - Parse and understand ONNX models
- ⚡ **Burn** - High-performance tensor operations
- 🎮 **wgpu** - Cross-platform GPU computing
- 🖼️ **Image** - Image loading and processing
- 🎯 **Clap** - Beautiful CLI interface
- 🛡️ **thiserror** - Error handling

---

## 🧪 Supported Operations

✅ Arithmetic operations (Add, Sub, Mul, Div, etc.)
✅ Activation functions (ReLU, GELU, Tanh, Sigmoid, etc.)
✅ Convolution layers (1D, 2D)
✅ Normalization layers (BatchNorm, LayerNorm)
✅ Pooling operations (MaxPool, AvgPool)
✅ Reduction operations (ReduceSum, ReduceMean, etc.)
✅ Shape operations (Reshape, Transpose, etc.)
✅ Matrix operations (MatMul, Gemm)
✅ Unary operations (Abs, Neg, Sqrt, Exp, Log, etc.)
✅ Comparison operations (Equal, Greater, Less, etc.)
✅ RNN/LSTM/GRU layers
✅ Audio operations
✅ And many more! 🎉

---

## ⚠️ Known Limitations

### 🔸 3D Convolution Support (Framework Limitation)

**Conv3d and ConvTranspose3d are not currently supported** ⛔

Due to limitations in the underlying Burn `DynTensor` framework (which only supports tensors up to rank-4), 3D convolution operations requiring rank-5 tensors cannot be executed. Models that use 3D convolutions (typically for video processing or medical imaging) will not run until either:
- The Burn framework is extended to support rank-5+ tensors, or
- An alternative tensor abstraction is implemented

**What works:**
- ✅ Conv1d (audio/time-series)
- ✅ Conv2d (image classification, detection)
- ✅ ConvTranspose1d/2d

**What doesn't work:**
- ⛔ Conv3d (video, volumetric data)
- ⛔ ConvTranspose3d

---

### 🔸 Large Model Parsing (Parser Issues)

**Some larger models may fail to parse due to bugs in the ONNX parser** 🐛

The `onnx-ir` library (used for parsing ONNX models) has known issues that can prevent loading certain models, particularly larger or more complex architectures.

**Workarounds:**
- Try simpler models with fewer operators
- Consider re-exporting models with minimal operator sets
- Use ONNX simplification tools to reduce model complexity

---

## 📖 Example Workflow

Here's a complete example using ResNet-18:

1. **Check devices** 👈
   ```bash
   cargo run --release -- devices
   ```

2. **Inspect model** 📋
   ```bash
   cargo run --release -- info test_data/resnet18.onnx
   ```

3. **Run inference** 🔮
   ```bash
   cargo run --release -- infer test_data/resnet18.onnx \
     --input test_data/sample.jpg \
     --labels test_data/imagenet_labels.txt \
     --top 5
   ```

---

## 🧑‍💻 Development

### Run tests

```bash
cargo test
```

### Build documentation

```bash
cargo doc --open
```

### Format code

```bash
cargo fmt
```

---

## 🤝 Contributing

We love contributions! 🎉

1. Fork the repository 🍴
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 🚀

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 💪 The [Burn](https://burn.dev/) team for the amazing deep learning framework
- 🎮 The [wgpu](https://github.com/gfx-rs/wgpu) team for the incredible GPU abstraction
- 📦 The [ONNX](https://onnx.ai/) community for the open model format
- 🦀 The Rust community for the awesome language and ecosystem

---

## 📧 Contact

For questions, suggestions, or just to say hi 👋:
- Open an issue on GitHub
- Reach out via Discussions

---

<div align="center">

Made with ❤️ and 🦀

**[⬆ Back to Top](#-nnx---neural-network-executor)**

</div>