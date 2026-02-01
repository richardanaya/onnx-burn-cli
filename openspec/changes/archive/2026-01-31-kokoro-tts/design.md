## Context

nnx is a runtime ONNX interpreter using `onnx-ir` (0.20.1) for parsing and Burn tensors for execution. All tensors are stored as `DynTensor<B>` which wraps rank-1 through rank-4 tensors internally as rank-4 for uniform handling. The dispatcher in `src/ops/mod.rs` routes `Node` variants to operator implementations.

The Kokoro-82M TTS model (82M parameters, 2,463 nodes, 49 unique operators) requires 7 operators not currently implemented. Of these, LSTM is the most complex (recurrent with gates), while others are relatively straightforward.

**Constraints:**
- DynTensor only supports up to rank-4 (no rank-5+ tensors)
- All tensor data stored as f32 internally
- Must follow existing operator pattern: `fn op<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()>`

## Goals / Non-Goals

**Goals:**
- Implement all 7 operators required by Kokoro-82M
- Enable nnx to successfully execute Kokoro TTS inference
- Follow existing codebase patterns for consistency
- Update CLI supported ops lists

**Non-Goals:**
- Full LSTM optimization (basic implementation is acceptable)
- GPU-optimized STFT kernels (CPU fallback acceptable for now)
- Supporting all STFT window functions (Hann only initially)
- Audio I/O or voice file handling (separate concern)

## Decisions

### 1. Operator Module Organization

**Decision:** Group new operators by category into existing or new modules.

| Operator | Module | Rationale |
|----------|--------|-----------|
| LSTM | `src/ops/rnn.rs` (new) | RNN ops are distinct category |
| Range | `src/ops/sequence.rs` (new) | Sequence generation ops |
| And | `src/ops/comparison.rs` (existing) | Logical ops extend comparison |
| ScatterND | `src/ops/advanced.rs` (existing) | Data manipulation like Gather |
| ConvTranspose | `src/ops/conv.rs` (existing) | Extends existing conv ops |
| Atan | `src/ops/unary.rs` (existing) | Trig function like Sin/Cos |
| STFT | `src/ops/audio.rs` (new) | Audio-specific transforms |

**Alternatives considered:**
- Single `src/ops/kokoro.rs` module: Rejected - violates separation of concerns
- All in `src/ops/advanced.rs`: Rejected - would make file too large

### 2. LSTM Implementation Strategy

**Decision:** Implement LSTM using explicit gate computations rather than Burn's RNN module.

```rust
// Explicit gate computation for clarity and control
let gates = matmul(x, W) + matmul(h, R) + bias;
let [i, o, f, c] = split_gates(gates);
let i = sigmoid(i);
let f = sigmoid(f);
let c_new = f * c_prev + i * tanh(c);
let h_new = o * tanh(c_new);
```

**Rationale:**
- ONNX LSTM has specific gate ordering (iofc) that may differ from Burn's
- Need to handle bidirectional and multi-layer variants
- Explicit implementation allows debugging gate values

**Alternatives considered:**
- Use Burn's LSTM module: Rejected - gate ordering mismatch concerns
- Delegate to external RNN crate: Rejected - adds dependency

### 3. STFT Implementation

**Decision:** Implement STFT using manual DFT computation on CPU.

**Rationale:**
- Burn doesn't have built-in FFT operations
- Kokoro uses STFT only once (1 node), so performance is not critical
- Can optimize later if needed

**Alternatives considered:**
- Use `rustfft` crate: Considered for future optimization
- GPU FFT via wgpu compute shaders: Too complex for initial implementation

### 4. ConvTranspose Handling

**Decision:** Verify existing ConvTranspose1d/2d implementations and add generic ConvTranspose dispatcher.

The ONNX `ConvTranspose` node may not specify dimensionality explicitly. We'll detect from input tensor rank:
- Rank 3 input → ConvTranspose1d
- Rank 4 input → ConvTranspose2d

## Risks / Trade-offs

**[LSTM Performance]** → Explicit gate computation may be slower than optimized RNN kernels.
- Mitigation: Acceptable for initial implementation; profile and optimize if bottleneck.

**[STFT Accuracy]** → Manual DFT may have numerical precision differences vs optimized FFT.
- Mitigation: Validate against reference implementation; use f64 intermediate if needed.

**[ConvTranspose Compatibility]** → Existing implementation may not handle all ONNX attributes.
- Mitigation: Test with Kokoro model specifically; add missing attribute handling as needed.

**[DynTensor Rank Limitation]** → Some operators might produce higher-rank outputs.
- Mitigation: LSTM hidden states are typically rank-3; verify Kokoro shapes fit within rank-4.
