## Why

The onnx-burn-cli currently supports only ~12 of ~110 ONNX operators defined in onnx-ir. This limits the CLI to running basic CNN models like ResNet. To support the broader ecosystem of ONNX models (transformers, object detection, RNNs, generative models), we need comprehensive operator coverage.

## What Changes

- Implement ~90 additional ONNX operators across all categories:
  - **Unary math ops**: abs, neg, sqrt, exp, log, ceil, floor, round, sign, reciprocal, sin, cos, tan, sinh, cosh, erf
  - **Activation functions**: tanh, gelu, leaky_relu, hard_sigmoid, hard_swish, prelu, log_softmax
  - **Shape manipulation**: squeeze, unsqueeze, transpose, concat, split, slice, expand, tile, pad, shape, size
  - **Reduction ops**: reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, argmax, argmin
  - **Binary ops**: pow, max, min, mean, sum, mod
  - **Comparison ops**: equal, greater, less, greater_equal, less_equal, is_inf, is_nan
  - **Linear algebra**: matmul, gemm
  - **Convolution variants**: conv1d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d
  - **Pooling variants**: avg_pool1d, avg_pool2d, max_pool1d
  - **Normalization**: layer_norm, instance_norm, group_norm
  - **Data manipulation**: gather, gather_elements, one_hot, where, topk, nonzero, cumsum
  - **Type conversion**: cast, constant_of_shape
  - **Advanced**: resize, grid_sample, depth_to_space, space_to_depth, lstm, attention
- Complete the 9 currently stubbed operators (matmul, gemm, tanh, concat, transpose, squeeze, unsqueeze, clip, cast)
- Organize operators into logical modules by category

## Capabilities

### New Capabilities
- `unary-math-ops`: Single-input mathematical operations (abs, neg, sqrt, exp, log, trig, etc.)
- `shape-ops`: Tensor shape manipulation (squeeze, unsqueeze, transpose, concat, split, etc.)
- `reduction-ops`: Reduction operations along axes (sum, mean, max, argmax, etc.)
- `comparison-ops`: Element-wise comparison operations returning bool tensors
- `pooling-ops`: Additional pooling variants (avg_pool1d, avg_pool2d, max_pool1d)
- `conv-ops`: Additional convolution variants (conv1d, conv3d, transposed convolutions)
- `normalization-ops`: Additional normalization layers (layer_norm, instance_norm, group_norm)
- `advanced-ops`: Complex operators (gather, topk, resize, lstm, attention)

### Modified Capabilities
(none - this is additive)

## Impact

- **Code**: New operator implementations in `src/ops/` modules, updates to dispatcher in `src/ops/mod.rs`
- **APIs**: No API changes - operators are internal implementation
- **Dependencies**: No new dependencies - all operators use existing burn tensor operations
- **Testing**: Each operator category should be tested with appropriate ONNX models
