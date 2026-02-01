## Context

We're building a CLI tool (`nnx`) that runs ONNX models at runtime using Burn's tensor operations with the wgpu GPU backend. This is a greenfield Rust project with no existing codebase.

Key constraints:
- No wonnx dependencies - use `onnx-ir` (from Burn ecosystem) for ONNX parsing
- Runtime model loading (not compile-time code generation like burn-import)
- Target wgpu backend for GPU acceleration
- Match core wonnx-cli functionality for devices, info, and infer commands

## Goals / Non-Goals

**Goals:**
- Load and execute arbitrary ONNX models at runtime
- Support common CNN operators (Conv2d, BatchNorm, pooling, activations, linear)
- Provide useful CLI for image classification workflows
- Enable running models like ResNet, MobileNet without recompilation

**Non-Goals:**
- Full ONNX operator coverage (v0.1 targets ~30-40 ops)
- Training support (inference only)
- Text/NLP model support (no tokenization in v0.1)
- Quantized model support
- Dynamic shape inference
- Benchmark mode (can add later)

## Decisions

### 1. Use `onnx-ir` for ONNX parsing

**Decision**: Use the `onnx-ir` crate from the Burn ecosystem.

**Rationale**: 
- Already parses ONNX into typed `Node` enum with 209 variants
- Handles protobuf parsing, topological sorting, type inference
- Weights/tensors are accessible at runtime
- Same ecosystem as Burn, ensuring compatibility

**Alternatives considered**:
- Raw protobuf parsing with `prost`: More work, reinventing the wheel
- `onnx-pb`: Lower-level, less structured than onnx-ir

### 2. Dynamic tensor representation

**Decision**: Use a `DynTensor` wrapper that erases the rank parameter, storing tensors in a `HashMap<String, DynTensor>`.

**Rationale**:
- ONNX graphs have dynamic connectivity - can't know tensor ranks at compile time
- Need to store heterogeneous tensors (different ranks) in one collection
- Trade-off: Lose compile-time rank checking, gain runtime flexibility

**Alternatives considered**:
- Enum with variants for each rank (Tensor1, Tensor2, etc.): Verbose, limited
- Always use 4D tensors with reshape: Wasteful, semantically wrong

### 3. Operator dispatch via match

**Decision**: Large `match` statement over `onnx_ir::Node` variants, dispatching to typed executor functions.

**Rationale**:
- Exhaustive matching ensures we handle or explicitly skip all ops
- Clear error messages for unsupported ops
- Easy to add new ops incrementally

**Implementation sketch**:
```rust
fn execute_node<B: Backend>(node: &Node, values: &mut ValueStore<B>) -> Result<()> {
    match node {
        Node::Add(n) => ops::add(n, values),
        Node::Relu(n) => ops::relu(n, values),
        Node::Conv2d(n) => ops::conv2d(n, values),
        // ...
        _ => Err(Error::UnsupportedOp(node.name().to_string())),
    }
}
```

### 4. Project structure

**Decision**: Single binary crate with modules for CLI, runtime, and ops.

```
src/
├── main.rs           # Entry point
├── cli/              # Clap command definitions
│   ├── mod.rs
│   ├── devices.rs
│   ├── info.rs
│   └── infer.rs
├── runtime/          # Execution engine
│   ├── mod.rs
│   ├── executor.rs   # Graph walker
│   ├── value_store.rs
│   └── tensor.rs     # DynTensor wrapper
├── ops/              # Op implementations
│   ├── mod.rs
│   ├── arithmetic.rs
│   ├── activation.rs
│   ├── conv.rs
│   └── ...
└── input/            # Input processing
    ├── mod.rs
    └── image.rs
```

### 5. Operator implementation priority

**Decision**: Implement ops in phases based on target model requirements.

**Phase 1** (MNIST/basic CNNs): Add, Relu, Conv2d, MaxPool2d, Flatten, Reshape, MatMul, Gemm, Softmax, Constant
**Phase 2** (ResNet): BatchNormalization, GlobalAveragePool, Concat, Transpose, Squeeze, Unsqueeze
**Phase 3** (broader): More activations, normalization variants, shape ops

## Risks / Trade-offs

**[Risk] Operator coverage gaps** → Start with well-known models (MNIST, ResNet). `nnx info` will show required ops so users know before running.

**[Risk] Performance vs wonnx** → Accept that interpreted execution may be slower than wonnx's optimized kernels. Burn's JIT should help. Can profile later.

**[Risk] Dynamic tensor overhead** → The DynTensor wrapper adds indirection. Acceptable for v0.1; can optimize hot paths later.

**[Trade-off] No dynamic shapes** → v0.1 assumes fixed input shapes. Simplifies implementation but limits some models.

**[Trade-off] Float32 only** → No fp16/int8 support initially. Covers most use cases, simplifies tensor handling.
