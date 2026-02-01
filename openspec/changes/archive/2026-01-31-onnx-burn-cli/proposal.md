## Why

Build a Rust CLI tool (`nnx`) for GPU-accelerated ONNX inference using Burn with the wgpu backend. This provides a pure-Rust alternative to wonnx-cli that leverages Burn's tensor operations for runtime ONNX execution, enabling users to run arbitrary ONNX models from the command line without compile-time model embedding.

## What Changes

- New CLI tool `nnx` with three subcommands:
  - `nnx devices` - List available GPU devices via wgpu
  - `nnx info <model>` - Display model metadata (inputs, outputs, operators used)
  - `nnx infer <model>` - Run inference with support for image inputs, labels, and top-k results
- Runtime ONNX interpreter using `onnx-ir` for parsing and Burn tensors for execution
- Support for common neural network operators (Conv2d, MatMul, BatchNorm, activations, pooling, etc.)
- Image input preprocessing for vision models

## Capabilities

### New Capabilities

- `cli-interface`: Command-line interface using clap with devices, info, and infer subcommands
- `device-discovery`: GPU device enumeration and selection via wgpu
- `model-inspection`: ONNX model parsing and metadata extraction (inputs, outputs, ops)
- `runtime-execution`: Runtime ONNX graph interpretation mapping onnx-ir nodes to Burn tensor ops
- `input-processing`: Image loading and preprocessing for model inputs

### Modified Capabilities

(none - greenfield project)

## Impact

- **Dependencies**: burn, burn-wgpu, onnx-ir, clap, image
- **New crate**: `nnx` binary crate
- **Target models**: ResNet, MobileNet, and similar CNN architectures for image classification
- **Operator coverage**: ~30-40 core ONNX operators for v0.1
