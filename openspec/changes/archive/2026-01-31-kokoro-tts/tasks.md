## 1. RNN Operations (src/ops/rnn.rs)

- [x] 1.1 Create src/ops/rnn.rs module file
- [x] 1.2 Implement LSTM forward pass with gate computations
- [x] 1.3 Handle LSTM initial hidden/cell state defaults
- [x] 1.4 Support bidirectional LSTM (direction attribute)
- [ ] 1.5 Support sequence_lens for variable length batches
- [x] 1.6 Wire up rnn module in src/ops/mod.rs dispatcher

## 2. Sequence Operations (src/ops/sequence.rs)

- [x] 2.1 Create src/ops/sequence.rs module file
- [x] 2.2 Implement Range operator (start, limit, delta)
- [x] 2.3 Handle Range with negative delta
- [x] 2.4 Handle Range with empty output
- [x] 2.5 Wire up sequence module in dispatcher

## 3. Logical Operations (extend src/ops/comparison.rs)

- [x] 3.1 Implement And operator with broadcasting
- [x] 3.2 Wire up And in dispatcher

## 4. Scatter Operations (extend src/ops/advanced.rs)

- [x] 4.1 Implement ScatterND operator basic functionality
- [x] 4.2 Handle ScatterND multi-dimensional indices
- [x] 4.3 Support ScatterND reduction modes (add, mul, max, min)
- [x] 4.4 Wire up ScatterND in dispatcher

## 5. Trigonometric Operations (extend src/ops/unary.rs)

- [x] 5.1 Implement Atan operator
- [x] 5.2 Wire up Atan in dispatcher

## 6. Audio Operations (src/ops/audio.rs)

- [x] 6.1 Create src/ops/audio.rs module file
- [x] 6.2 Implement DFT helper function
- [x] 6.3 Implement STFT operator with windowing
- [x] 6.4 Handle STFT onesided attribute
- [x] 6.5 Support batched STFT inputs
- [x] 6.6 Wire up audio module in dispatcher

## 7. ConvTranspose Fix (src/ops/conv.rs)

- [x] 7.1 Verify ConvTranspose dispatching from generic ONNX node
- [x] 7.2 Add ConvTranspose rank detection (1d vs 2d)
- [ ] 7.3 Test ConvTranspose with Kokoro model nodes

## 8. Update Supported Ops Lists

- [x] 8.1 Update is_supported_op() in src/cli/info.rs with new operators
- [x] 8.2 Update is_supported_op() in src/cli/infer.rs with new operators

## 9. Validation

- [ ] 9.1 Run nnx info on Kokoro model to verify all operators recognized
- [ ] 9.2 Attempt full Kokoro model execution (may require debugging)
- [ ] 9.3 Validate LSTM output shapes match expected
- [ ] 9.4 Validate STFT output against reference implementation
