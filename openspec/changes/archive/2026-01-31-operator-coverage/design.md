## Context

We're building a runtime ONNX interpreter using onnx-ir for parsing and Burn tensors for execution. Currently ~12 operators are implemented, ~9 are stubbed, and ~89 remain. The existing codebase has established patterns:

- `src/ops/mod.rs`: Central dispatcher with `match` on `onnx_ir::ir::Node` variants
- `src/ops/*.rs`: Operator implementations grouped by category (arithmetic, activation, conv, linear, shape)
- `DynTensor<B>`: Rank-erased tensor wrapper for runtime shape handling
- `ValueStore<B>`: HashMap-based storage for intermediate values

Constraints:
- Must work with burn-wgpu backend (GPU execution)
- onnx-ir defines the node structures we must match
- All operators should follow existing patterns for consistency

## Goals / Non-Goals

**Goals:**
- Implement all ~90 remaining operators that onnx-ir supports
- Maintain consistent code patterns across operator categories
- Enable running transformer models (needs MatMul, LayerNorm, Gather, etc.)
- Enable running object detection models (needs Concat, Resize, TopK, etc.)
- Enable running sequence models (needs LSTM, Squeeze/Unsqueeze, etc.)

**Non-Goals:**
- Training support (inference only)
- Custom operator extensions
- Performance optimization (focus on correctness first)
- Control flow operators (If, Loop, Scan) - rare in exported models

## Decisions

### 1. Organize operators by implementation pattern, not ONNX category

**Decision**: Group implementation work by code pattern similarity.

**Rationale**: Operators with similar patterns can be implemented together efficiently:
- "One-liner math" (abs, neg, sqrt, etc.) - all call `tensor.method()`
- "Binary with broadcast" (pow, max, min) - same pattern as existing Add/Sub/Mul/Div
- "Shape manipulators" (squeeze, transpose, concat) - read axis config, call reshape/permute

**Alternatives considered**:
- Group by ONNX spec category → less efficient, mixed complexity per batch
- Alphabetical → no logical grouping benefit

### 2. Create new module files for each operator category

**Decision**: Add new files to `src/ops/`:
- `unary.rs` - single-input math ops
- `comparison.rs` - comparison ops returning bool tensors
- `reduction.rs` - reduction ops with axis parameters
- `normalization.rs` - layer_norm, instance_norm, group_norm
- `pooling.rs` - avg_pool variants (extend existing conv.rs or separate)
- `advanced.rs` - complex ops (gather, topk, resize, etc.)

**Rationale**: Keeps files focused and manageable. Existing pattern already separates arithmetic.rs, activation.rs, conv.rs, linear.rs, shape.rs.

### 3. Use Burn's tensor API directly where possible

**Decision**: Map ONNX ops to Burn tensor methods: `tensor.abs()`, `tensor.transpose()`, `activation::gelu()`, etc.

**Rationale**: Burn already implements most tensor operations. Only complex ops (LSTM, Attention, Resize) need custom logic.

**Alternatives considered**:
- Implement ops from scratch → unnecessary, Burn has them
- Use burn-import → designed for compile-time, not runtime interpretation

### 4. Handle DynTensor rank polymorphism with match statements

**Decision**: Continue using `match input_dyn.rank() { 1 => ..., 2 => ..., 4 => ... }` pattern.

**Rationale**: Rust's type system requires compile-time known ranks. Match statement is verbose but explicit and correct.

**Alternatives considered**:
- Macro to reduce boilerplate → adds complexity, harder to debug
- Always use rank-4 internally → loses shape information, broadcasting issues

### 5. Defer control flow operators (If, Loop, Scan)

**Decision**: Mark If, Loop, Scan as unsupported. Focus on data-flow operators.

**Rationale**: Control flow requires subgraph execution, significantly more complex. Rarely needed for standard model inference - most models are static graphs.

## Risks / Trade-offs

**[Risk] Some operators may have subtle ONNX semantics** → Test each operator with real ONNX models or ONNX conformance tests. Read ONNX spec when implementing.

**[Risk] Burn may not have direct equivalents for all ops** → For missing ops, implement using lower-level tensor operations. Flag these for review.

**[Risk] Large scope (~90 operators)** → Batch implementation by pattern. Track progress in tasks.md. Prioritize operators needed for specific model families.

**[Risk] DynTensor rank handling is verbose** → Accept verbosity for correctness. Consider macro extraction later if patterns stabilize.

**[Trade-off] Correctness over performance** → Initial implementation focuses on working correctly. Performance optimization is a separate future effort.
