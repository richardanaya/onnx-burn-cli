use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;
use onnx_ir::node::lstm::LstmDirection;

/// LSTM operator - Long Short-Term Memory recurrent neural network
///
/// Inputs:
///   0: X      - input sequence [seq_length, batch_size, input_size]
///   1: W      - weight tensor [num_directions, 4*hidden_size, input_size]
///   2: R      - recurrence weight [num_directions, 4*hidden_size, hidden_size]
///   3: B      - bias (optional) [num_directions, 8*hidden_size]
///   4: sequence_lens - (optional, not supported)
///   5: initial_h - initial hidden state (optional) [num_directions, batch_size, hidden_size]
///   6: initial_c - initial cell state (optional) [num_directions, batch_size, hidden_size]
///   7: P      - peephole weights (optional, not supported)
///
/// Outputs:
///   0: Y      - all hidden states [seq_length, num_directions, batch_size, hidden_size]
///   1: Y_h    - final hidden state [num_directions, batch_size, hidden_size]
///   2: Y_c    - final cell state [num_directions, batch_size, hidden_size]
///
/// Gate order in weights: i (input), o (output), f (forget), c (cell)
pub fn lstm<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Lstm(n) = node {
        let config = &n.config;
        let hidden_size = config.hidden_size;
        let num_directions = config.direction.num_directions();

        // Get input tensor X: [seq_length, batch_size, input_size]
        let x_name = &n.inputs[0].name;
        let x_dyn = values
            .get(x_name)
            .ok_or_else(|| anyhow!("LSTM: input X '{}' not found", x_name))?;

        let x_shape = x_dyn.shape().to_vec();
        if x_shape.len() != 3 {
            return Err(anyhow!(
                "LSTM: X must be rank 3 [seq, batch, input], got rank {}",
                x_shape.len()
            ));
        }

        let seq_length = x_shape[0];
        let batch_size = x_shape[1];
        let input_size = x_shape[2];

        // Get X data as flat vector
        let x_4d = x_dyn.as_rank4();
        let x_data: Vec<f32> = x_4d
            .to_data()
            .to_vec()
            .map_err(|e| anyhow!("LSTM: cannot get X data: {:?}", e))?;

        // Get weight tensor W: [num_directions, 4*hidden_size, input_size]
        let w_data = get_weight_data(&n.inputs[1], values)?;
        // Get recurrence weight R: [num_directions, 4*hidden_size, hidden_size]
        let r_data = get_weight_data(&n.inputs[2], values)?;

        // Get optional bias B: [num_directions, 8*hidden_size]
        // Format: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c] for each direction
        let b_data = if config.has_bias && n.inputs.len() > 3 {
            Some(get_weight_data(&n.inputs[3], values)?)
        } else {
            None
        };

        // Get optional initial hidden state: [num_directions, batch_size, hidden_size]
        let initial_h = if config.has_initial_h && n.inputs.len() > 5 {
            let h_name = &n.inputs[5].name;
            if let Some(h_dyn) = values.get(h_name) {
                let h_4d = h_dyn.as_rank4();
                Some(
                    h_4d.to_data()
                        .to_vec()
                        .map_err(|e| anyhow!("LSTM: cannot get initial_h: {:?}", e))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Get optional initial cell state: [num_directions, batch_size, hidden_size]
        let initial_c = if config.has_initial_c && n.inputs.len() > 6 {
            let c_name = &n.inputs[6].name;
            if let Some(c_dyn) = values.get(c_name) {
                let c_4d = c_dyn.as_rank4();
                Some(
                    c_4d.to_data()
                        .to_vec()
                        .map_err(|e| anyhow!("LSTM: cannot get initial_c: {:?}", e))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Allocate output tensors
        // Y: [seq_length, num_directions, batch_size, hidden_size]
        let mut y_data = vec![0.0f32; seq_length * num_directions * batch_size * hidden_size];
        // Y_h: [num_directions, batch_size, hidden_size]
        let mut y_h_data = vec![0.0f32; num_directions * batch_size * hidden_size];
        // Y_c: [num_directions, batch_size, hidden_size]
        let mut y_c_data = vec![0.0f32; num_directions * batch_size * hidden_size];

        // Process each direction
        for dir in 0..num_directions {
            let is_reverse = match &config.direction {
                LstmDirection::Forward => false,
                LstmDirection::Reverse => true,
                LstmDirection::Bidirectional => dir == 1,
            };

            // Extract weights for this direction
            // W: [4*hidden_size, input_size] for this direction
            let w_offset = dir * 4 * hidden_size * input_size;
            let w_dir = &w_data[w_offset..w_offset + 4 * hidden_size * input_size];

            // R: [4*hidden_size, hidden_size] for this direction
            let r_offset = dir * 4 * hidden_size * hidden_size;
            let r_dir = &r_data[r_offset..r_offset + 4 * hidden_size * hidden_size];

            // Bias: [8*hidden_size] for this direction (Wb + Rb)
            let bias = if let Some(ref b) = b_data {
                let b_offset = dir * 8 * hidden_size;
                Some(&b[b_offset..b_offset + 8 * hidden_size])
            } else {
                None
            };

            // Initialize hidden and cell states
            let mut h_t = vec![0.0f32; batch_size * hidden_size];
            let mut c_t = vec![0.0f32; batch_size * hidden_size];

            if let Some(ref h_init) = initial_h {
                let h_offset = dir * batch_size * hidden_size;
                h_t.copy_from_slice(&h_init[h_offset..h_offset + batch_size * hidden_size]);
            }
            if let Some(ref c_init) = initial_c {
                let c_offset = dir * batch_size * hidden_size;
                c_t.copy_from_slice(&c_init[c_offset..c_offset + batch_size * hidden_size]);
            }

            // Process sequence
            let time_steps: Vec<usize> = if is_reverse {
                (0..seq_length).rev().collect()
            } else {
                (0..seq_length).collect()
            };

            for &t in time_steps.iter() {
                // Get input at time t: [batch_size, input_size]
                let x_t_offset = t * batch_size * input_size;
                let x_t = &x_data[x_t_offset..x_t_offset + batch_size * input_size];

                // Compute gates for all batches
                // gates = x_t @ W^T + h_t @ R^T + bias
                // Gate order: i, o, f, c (each of size hidden_size)

                let mut gates = vec![0.0f32; batch_size * 4 * hidden_size];

                // Compute x_t @ W^T (input contribution)
                for b in 0..batch_size {
                    for g in 0..(4 * hidden_size) {
                        let mut sum = 0.0f32;
                        for i in 0..input_size {
                            // W is [4*hidden_size, input_size], so W[g, i]
                            sum += x_t[b * input_size + i] * w_dir[g * input_size + i];
                        }
                        gates[b * 4 * hidden_size + g] = sum;
                    }
                }

                // Add h_t @ R^T (recurrent contribution)
                for b in 0..batch_size {
                    for g in 0..(4 * hidden_size) {
                        let mut sum = 0.0f32;
                        for h in 0..hidden_size {
                            // R is [4*hidden_size, hidden_size], so R[g, h]
                            sum += h_t[b * hidden_size + h] * r_dir[g * hidden_size + h];
                        }
                        gates[b * 4 * hidden_size + g] += sum;
                    }
                }

                // Add bias if present
                if let Some(bias) = bias {
                    for b in 0..batch_size {
                        for g in 0..(4 * hidden_size) {
                            // Wb is first 4*hidden_size, Rb is second 4*hidden_size
                            gates[b * 4 * hidden_size + g] += bias[g] + bias[4 * hidden_size + g];
                        }
                    }
                }

                // Apply activations and compute new states
                // Gate order in ONNX: i, o, f, c
                for b in 0..batch_size {
                    let gate_base = b * 4 * hidden_size;

                    for h in 0..hidden_size {
                        // Extract gate values
                        let i_gate = gates[gate_base + h]; // input gate
                        let o_gate = gates[gate_base + hidden_size + h]; // output gate
                        let f_gate = gates[gate_base + 2 * hidden_size + h]; // forget gate
                        let c_gate = gates[gate_base + 3 * hidden_size + h]; // cell gate

                        // Apply activations (default: sigmoid for gates, tanh for cell)
                        let i = sigmoid(i_gate);
                        let o = sigmoid(o_gate);
                        let f = sigmoid(f_gate);
                        let c_candidate = tanh(c_gate);

                        // Update cell state: c_t = f * c_{t-1} + i * c_candidate
                        let c_prev = c_t[b * hidden_size + h];
                        let c_new = f * c_prev + i * c_candidate;

                        // Apply cell clipping if configured
                        let c_new = if let Some(clip) = config.clip {
                            c_new.max(-clip).min(clip)
                        } else {
                            c_new
                        };

                        c_t[b * hidden_size + h] = c_new;

                        // Update hidden state: h_t = o * tanh(c_t)
                        let h_new = o * tanh(c_new);
                        h_t[b * hidden_size + h] = h_new;
                    }
                }

                // Store hidden state in Y output
                // Y: [seq_length, num_directions, batch_size, hidden_size]
                // For reverse direction, we still store in original time order
                let y_t_offset =
                    t * num_directions * batch_size * hidden_size + dir * batch_size * hidden_size;
                y_data[y_t_offset..y_t_offset + batch_size * hidden_size].copy_from_slice(&h_t);
            }

            // Store final states
            let h_offset = dir * batch_size * hidden_size;
            y_h_data[h_offset..h_offset + batch_size * hidden_size].copy_from_slice(&h_t);
            y_c_data[h_offset..h_offset + batch_size * hidden_size].copy_from_slice(&c_t);
        }

        // Create output tensors
        // Output Y if present
        if !n.outputs.is_empty() && !n.outputs[0].name.is_empty() {
            let y_tensor: Tensor<B, 4> = Tensor::from_data(
                TensorData::new(
                    y_data,
                    [seq_length, num_directions, batch_size, hidden_size],
                ),
                device,
            );
            values.insert(n.outputs[0].name.clone(), DynTensor::from_rank4(y_tensor));
        }

        // Output Y_h if present
        if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
            let y_h_tensor: Tensor<B, 3> = Tensor::from_data(
                TensorData::new(y_h_data, [num_directions, batch_size, hidden_size]),
                device,
            );
            values.insert(n.outputs[1].name.clone(), DynTensor::from_rank3(y_h_tensor));
        }

        // Output Y_c if present
        if n.outputs.len() > 2 && !n.outputs[2].name.is_empty() {
            let y_c_tensor: Tensor<B, 3> = Tensor::from_data(
                TensorData::new(y_c_data, [num_directions, batch_size, hidden_size]),
                device,
            );
            values.insert(n.outputs[2].name.clone(), DynTensor::from_rank3(y_c_tensor));
        }

        Ok(())
    } else {
        Err(anyhow!("Not an Lstm node"))
    }
}

/// Helper to get weight data from input (either from value store or constant)
fn get_weight_data<B: Backend>(
    input: &onnx_ir::ir::Argument,
    values: &ValueStore<B>,
) -> Result<Vec<f32>> {
    // Try to get from constant data first
    if let Some(tensor_data) = input.value() {
        let slice = tensor_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("LSTM: cannot convert weight to f32: {:?}", e))?;
        return Ok(slice.to_vec());
    }

    // Try to get from value store
    if let Some(dyn_tensor) = values.get(&input.name) {
        let tensor_4d = dyn_tensor.as_rank4();
        let data = tensor_4d.to_data();
        let slice: Vec<f32> = data
            .to_vec()
            .map_err(|e| anyhow!("LSTM: cannot get weight data: {:?}", e))?;
        return Ok(slice);
    }

    Err(anyhow!("LSTM: weight tensor '{}' not found", input.name))
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation
#[inline]
fn tanh(x: f32) -> f32 {
    x.tanh()
}
