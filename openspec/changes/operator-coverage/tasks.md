## 1. Unary Math Operations (src/ops/unary.rs)

- [x] 1.1 Create src/ops/unary.rs module file
- [x] 1.2 Implement Abs operator (tensor.abs())
- [x] 1.3 Implement Neg operator (tensor.neg())
- [x] 1.4 Implement Sqrt operator (tensor.sqrt())
- [x] 1.5 Implement Exp operator (tensor.exp())
- [x] 1.6 Implement Log operator (tensor.log())
- [x] 1.7 Implement Ceil operator (tensor.ceil())
- [x] 1.8 Implement Floor operator (tensor.floor())
- [x] 1.9 Implement Round operator
- [x] 1.10 Implement Sign operator
- [x] 1.11 Implement Reciprocal operator
- [x] 1.12 Implement Sin operator (tensor.sin())
- [x] 1.13 Implement Cos operator (tensor.cos())
- [x] 1.14 Implement Tan operator
- [x] 1.15 Implement Sinh operator
- [x] 1.16 Implement Cosh operator
- [x] 1.17 Implement Erf operator
- [x] 1.18 Add unary module to src/ops/mod.rs and wire up dispatcher

## 2. Additional Activation Functions (src/ops/activation.rs)

- [x] 2.1 Implement Tanh operator (activation::tanh)
- [x] 2.2 Implement Gelu operator (activation::gelu)
- [x] 2.3 Implement LeakyRelu operator with alpha config
- [x] 2.4 Implement HardSigmoid operator
- [x] 2.5 Implement HardSwish operator
- [x] 2.6 Implement Prelu operator with learned alpha
- [x] 2.7 Implement LogSoftmax operator
- [x] 2.8 Wire up new activations in dispatcher

## 3. Trivial Pass-through Operations

- [x] 3.1 Implement Identity operator (return input unchanged)
- [x] 3.2 Implement Dropout operator (no-op in inference)
- [x] 3.3 Implement Clip operator (tensor.clamp(min, max))

## 4. Shape Manipulation Operations (src/ops/shape.rs)

- [x] 4.1 Implement Squeeze operator
- [x] 4.2 Implement Unsqueeze operator
- [x] 4.3 Implement Transpose operator with permutation
- [x] 4.4 Implement Concat operator (Tensor::cat)
- [ ] 4.5 Implement Split operator
- [ ] 4.6 Implement Slice operator
- [ ] 4.7 Implement Expand operator
- [ ] 4.8 Implement Tile operator
- [ ] 4.9 Implement Pad operator
- [x] 4.10 Implement Shape operator (return shape as tensor)
- [x] 4.11 Implement Size operator (return total elements)
- [x] 4.12 Wire up shape ops in dispatcher

## 5. Reduction Operations (src/ops/reduction.rs)

- [x] 5.1 Create src/ops/reduction.rs module file
- [x] 5.2 Implement ReduceSum operator (tensor.sum_dim)
- [x] 5.3 Implement ReduceMean operator (tensor.mean_dim)
- [x] 5.4 Implement ReduceMax operator (tensor.max_dim)
- [x] 5.5 Implement ReduceMin operator (tensor.min_dim)
- [x] 5.6 Implement ReduceProd operator
- [x] 5.7 Implement Argmax operator (tensor.argmax)
- [x] 5.8 Implement Argmin operator (tensor.argmin)
- [x] 5.9 Add keepdims handling for all reduction ops
- [x] 5.10 Wire up reduction module in dispatcher

## 6. Comparison Operations (src/ops/comparison.rs)

- [x] 6.1 Create src/ops/comparison.rs module file
- [x] 6.2 Implement Equal operator (tensor.equal)
- [x] 6.3 Implement Greater operator (tensor.greater)
- [x] 6.4 Implement Less operator (tensor.lower)
- [x] 6.5 Implement GreaterOrEqual operator
- [x] 6.6 Implement LessOrEqual operator
- [x] 6.7 Implement IsInf operator
- [x] 6.8 Implement IsNaN operator
- [x] 6.9 Wire up comparison module in dispatcher

## 7. Binary Operations (extend src/ops/arithmetic.rs)

- [x] 7.1 Implement Pow operator (tensor.powf)
- [x] 7.2 Implement Max (element-wise) operator
- [x] 7.3 Implement Min (element-wise) operator
- [x] 7.4 Implement Mod operator
- [x] 7.5 Implement Sum (multi-input) operator
- [x] 7.6 Implement Mean (multi-input) operator

## 8. Pooling Operations (extend src/ops/conv.rs or new pooling.rs)

- [x] 8.1 Implement AvgPool2d operator (module::avg_pool2d)
- [x] 8.2 Implement AvgPool1d operator (module::avg_pool1d)
- [x] 8.3 Implement MaxPool1d operator (module::max_pool1d)
- [x] 8.4 Wire up pooling ops in dispatcher

## 9. Convolution Variants (extend src/ops/conv.rs)

- [x] 9.1 Implement Conv1d operator
- [ ] 9.2 Implement Conv3d operator
- [x] 9.3 Implement ConvTranspose2d operator
- [ ] 9.4 Implement ConvTranspose1d operator
- [ ] 9.5 Implement ConvTranspose3d operator
- [x] 9.6 Wire up conv variants in dispatcher

## 10. Normalization Operations (src/ops/normalization.rs)

- [x] 10.1 Create src/ops/normalization.rs module file
- [x] 10.2 Implement LayerNorm operator
- [x] 10.3 Implement InstanceNorm operator
- [x] 10.4 Implement GroupNorm operator
- [x] 10.5 Wire up normalization module in dispatcher

## 11. Linear Algebra Operations (extend src/ops/linear.rs)

- [x] 11.1 Implement MatMul operator with batching support
- [x] 11.2 Implement Gemm operator (alpha*A*B + beta*C)
- [x] 11.3 Handle transA/transB flags in Gemm
- [x] 11.4 Wire up matmul/gemm in dispatcher

## 12. Advanced Data Operations (src/ops/advanced.rs)

- [ ] 12.1 Create src/ops/advanced.rs module file
- [ ] 12.2 Implement Gather operator
- [ ] 12.3 Implement GatherElements operator
- [ ] 12.4 Implement TopK operator
- [ ] 12.5 Implement NonZero operator
- [ ] 12.6 Implement Where operator
- [ ] 12.7 Implement OneHot operator
- [ ] 12.8 Implement CumSum operator
- [ ] 12.9 Wire up advanced module in dispatcher

## 13. Type Conversion Operations

- [ ] 13.1 Implement Cast operator (dtype conversion)
- [ ] 13.2 Implement ConstantOfShape operator

## 14. Spatial Transform Operations

- [ ] 14.1 Implement Resize operator with nearest neighbor mode
- [ ] 14.2 Implement Resize operator with bilinear mode
- [ ] 14.3 Implement DepthToSpace operator
- [ ] 14.4 Implement SpaceToDepth operator

## 15. Update Supported Ops List

- [ ] 15.1 Update is_supported_op() in src/cli/info.rs with all new operators
- [ ] 15.2 Update is_supported_op() in src/cli/infer.rs with all new operators

## 16. Testing

- [ ] 16.1 Test unary math ops with simple tensors
- [ ] 16.2 Test shape ops with various input shapes
- [ ] 16.3 Test reduction ops with keepdims variations
- [ ] 16.4 Test comparison ops return correct bool tensors
- [ ] 16.5 Test a transformer model (needs MatMul, LayerNorm, Gather)
- [ ] 16.6 Test an object detection model (needs Concat, Resize, TopK)
