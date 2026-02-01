use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;
use std::f32::consts::PI;

/// STFT operator - Short-Time Fourier Transform
///
/// Inputs:
///   0: signal      - input signal [batch_size, signal_length, 1] or [batch_size, signal_length]
///   1: frame_step  - step size between frames (scalar int64)
///   2: window      - window function (optional) [frame_length]
///   3: frame_length - length of each frame (optional, scalar int64, defaults to window length)
///
/// Output:
///   0: output      - STFT result [batch_size, num_frames, fft_size, 2] (real/imag)
///                    where fft_size = frame_length/2 + 1 if onesided, else frame_length
///
/// Attributes:
///   onesided - if 1 (default), return only the first half+1 of FFT bins
pub fn stft<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Stft(n) = node {
        let output_name = &n.outputs[0].name;

        // Get signal input
        let signal_name = &n.inputs[0].name;
        let signal_dyn = values
            .get(signal_name)
            .ok_or_else(|| anyhow!("STFT: signal '{}' not found", signal_name))?;

        let signal_shape = signal_dyn.shape().to_vec();
        let signal_4d = signal_dyn.as_rank4();
        let signal_data: Vec<f32> = signal_4d
            .to_data()
            .to_vec()
            .map_err(|e| anyhow!("STFT: cannot get signal data: {:?}", e))?;

        // Determine batch_size and signal_length
        let (batch_size, signal_length) = match signal_shape.len() {
            2 => (signal_shape[0], signal_shape[1]),
            3 => (signal_shape[0], signal_shape[1]),
            _ => {
                return Err(anyhow!(
                    "STFT: signal must be 2D or 3D, got {}D",
                    signal_shape.len()
                ))
            }
        };

        // Get frame_step
        let frame_step = get_scalar_int(&n.inputs[1], values)?;
        if frame_step <= 0 {
            return Err(anyhow!(
                "STFT: frame_step must be positive, got {}",
                frame_step
            ));
        }
        let frame_step = frame_step as usize;

        // Get window (optional)
        let window_data = if n.inputs.len() > 2 && !n.inputs[2].name.is_empty() {
            Some(get_tensor_data(&n.inputs[2], values)?)
        } else {
            None
        };

        // Get frame_length (optional, defaults to window length)
        let frame_length = if n.inputs.len() > 3 && !n.inputs[3].name.is_empty() {
            get_scalar_int(&n.inputs[3], values)? as usize
        } else if let Some(ref w) = window_data {
            w.len()
        } else {
            return Err(anyhow!(
                "STFT: either window or frame_length must be provided"
            ));
        };

        // Create default window (rectangular) if not provided
        let window = window_data.unwrap_or_else(|| vec![1.0f32; frame_length]);

        // Ensure window length matches frame_length
        if window.len() != frame_length {
            return Err(anyhow!(
                "STFT: window length {} doesn't match frame_length {}",
                window.len(),
                frame_length
            ));
        }

        // Get onesided attribute (default true)
        // Since StftNode is unsupported, we check raw attributes
        // For now, default to onesided=true which is the ONNX default
        let onesided = true;

        // Calculate output dimensions
        let num_frames = (signal_length - frame_length) / frame_step + 1;
        let fft_size = if onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };

        // Allocate output: [batch_size, num_frames, fft_size, 2]
        let mut output_data = vec![0.0f32; batch_size * num_frames * fft_size * 2];

        // Process each batch
        for b in 0..batch_size {
            let batch_offset = b * signal_length;

            // Process each frame
            for f in 0..num_frames {
                let frame_start = f * frame_step;

                // Extract and window the frame
                let mut frame = vec![0.0f32; frame_length];
                for i in 0..frame_length {
                    let signal_idx = batch_offset + frame_start + i;
                    frame[i] = signal_data[signal_idx] * window[i];
                }

                // Compute DFT for this frame
                let dft_result = compute_dft(&frame, onesided);

                // Store result
                let output_frame_offset = (b * num_frames + f) * fft_size * 2;
                for k in 0..fft_size {
                    output_data[output_frame_offset + k * 2] = dft_result[k * 2]; // real
                    output_data[output_frame_offset + k * 2 + 1] = dft_result[k * 2 + 1];
                    // imag
                }
            }
        }

        // Create output tensor
        let output_tensor: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(output_data, [batch_size, num_frames, fft_size, 2]),
            device,
        );
        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));

        Ok(())
    } else {
        Err(anyhow!("Not a Stft node"))
    }
}

/// Compute DFT (Discrete Fourier Transform) of a real signal
/// Returns interleaved real/imag pairs: [re0, im0, re1, im1, ...]
fn compute_dft(signal: &[f32], onesided: bool) -> Vec<f32> {
    let n = signal.len();
    let k_max = if onesided { n / 2 + 1 } else { n };

    let mut result = Vec::with_capacity(k_max * 2);

    for k in 0..k_max {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        for (t, &x) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            real += x * angle.cos();
            imag += x * angle.sin();
        }

        result.push(real);
        result.push(imag);
    }

    result
}

/// Helper to get scalar integer from input
fn get_scalar_int<B: Backend>(
    input: &onnx_ir::ir::Argument,
    values: &ValueStore<B>,
) -> Result<i64> {
    // Try constant data first
    if let Some(tensor_data) = input.value() {
        if let Ok(slice) = tensor_data.as_slice::<i64>() {
            return Ok(slice[0]);
        }
        if let Ok(slice) = tensor_data.as_slice::<i32>() {
            return Ok(slice[0] as i64);
        }
    }

    // Try value store
    if let Some(dyn_tensor) = values.get(&input.name) {
        let tensor_4d = dyn_tensor.as_rank4();
        let data = tensor_4d.to_data();
        let floats: Vec<f32> = data
            .to_vec()
            .map_err(|e| anyhow!("STFT: cannot get scalar: {:?}", e))?;
        return Ok(floats[0] as i64);
    }

    Err(anyhow!("STFT: scalar input '{}' not found", input.name))
}

/// Helper to get tensor data as f32 vector
fn get_tensor_data<B: Backend>(
    input: &onnx_ir::ir::Argument,
    values: &ValueStore<B>,
) -> Result<Vec<f32>> {
    // Try constant data first
    if let Some(tensor_data) = input.value() {
        if let Ok(slice) = tensor_data.as_slice::<f32>() {
            return Ok(slice.to_vec());
        }
    }

    // Try value store
    if let Some(dyn_tensor) = values.get(&input.name) {
        let tensor_4d = dyn_tensor.as_rank4();
        let data = tensor_4d.to_data();
        let floats: Vec<f32> = data
            .to_vec()
            .map_err(|e| anyhow!("STFT: cannot get tensor data: {:?}", e))?;
        return Ok(floats);
    }

    Err(anyhow!("STFT: tensor input '{}' not found", input.name))
}
