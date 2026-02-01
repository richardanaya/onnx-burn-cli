## Why

nnx currently supports ~100 ONNX operators but cannot run the Kokoro-82M TTS model, which requires 7 additional operators (LSTM, Range, And, ScatterND, ConvTranspose, Atan, STFT). Adding these operators will enable nnx to run state-of-the-art text-to-speech inference, demonstrating the runtime's capability beyond vision models.

## What Changes

- Add 7 new ONNX operators required by Kokoro TTS:
  - **LSTM**: Recurrent neural network layer (6 nodes in Kokoro)
  - **Range**: Generate sequence tensor (2 nodes)
  - **And**: Element-wise logical AND (1 node)
  - **ScatterND**: Scatter values into tensor by indices (1 node)
  - **ConvTranspose**: Transposed convolution (6 nodes, extends existing ConvTranspose1d/2d)
  - **Atan**: Arctangent trigonometric function (1 node)
  - **STFT**: Short-time Fourier transform for audio (1 node)
- Update `is_supported_op()` in CLI info/infer modules
- Add Kokoro model to test_data for validation

## Capabilities

### New Capabilities

- `rnn-ops`: LSTM and related recurrent neural network operators
- `sequence-ops`: Range and sequence generation operators
- `logical-ops`: And, Or, Not, Xor logical operators
- `scatter-ops`: ScatterND and ScatterElements tensor scatter operators
- `trig-ops`: Atan, Asin, Acos extended trigonometric operators
- `audio-ops`: STFT and audio processing operators

### Modified Capabilities

None - all changes are additive new operators.

## Impact

- **Code**: New operator implementations in `src/ops/` modules
- **CLI**: Updated supported ops lists in `src/cli/info.rs` and `src/cli/infer.rs`
- **Tests**: Kokoro model added to `test_data/kokoro/` for validation
- **Dependencies**: No new dependencies required (burn already has needed primitives)
