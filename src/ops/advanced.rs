use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;

/// Helper to get input tensor from value store or constant data
fn get_input_tensor<B: Backend>(
    input: &onnx_ir::ir::Argument,
    values: &ValueStore<B>,
    device: &B::Device,
) -> Result<DynTensor<B>> {
    // First try to get from value store
    if let Some(dyn_tensor) = values.get(&input.name) {
        return Ok(dyn_tensor.clone());
    }

    // Otherwise try to get from constant data
    if let Some(tensor_data) = input.value() {
        let slice = tensor_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert constant to f32: {:?}", e))?;
        let shape: Vec<usize> = tensor_data.shape.iter().map(|&d| d as usize).collect();

        match shape.len() {
            0 => {
                let data = TensorData::new(slice.to_vec(), [1]);
                let t: Tensor<B, 1> = Tensor::from_data(data, device);
                Ok(DynTensor::from_rank1(t))
            }
            1 => {
                let data = TensorData::new(slice.to_vec(), [shape[0]]);
                let t: Tensor<B, 1> = Tensor::from_data(data, device);
                Ok(DynTensor::from_rank1(t))
            }
            2 => {
                let data = TensorData::new(slice.to_vec(), [shape[0], shape[1]]);
                let t: Tensor<B, 2> = Tensor::from_data(data, device);
                Ok(DynTensor::from_rank2(t))
            }
            3 => {
                let data = TensorData::new(slice.to_vec(), [shape[0], shape[1], shape[2]]);
                let t: Tensor<B, 3> = Tensor::from_data(data, device);
                Ok(DynTensor::from_rank3(t))
            }
            4 => {
                let data =
                    TensorData::new(slice.to_vec(), [shape[0], shape[1], shape[2], shape[3]]);
                let t: Tensor<B, 4> = Tensor::from_data(data, device);
                Ok(DynTensor::from_rank4(t))
            }
            _ => Err(anyhow!("Unsupported tensor rank: {}", shape.len())),
        }
    } else {
        Err(anyhow!("Input tensor '{}' not found", input.name))
    }
}

/// Helper to get indices tensor (int64)
fn get_indices_tensor<B: Backend>(
    input: &onnx_ir::ir::Argument,
    values: &ValueStore<B>,
    _device: &B::Device,
) -> Result<(Vec<i64>, Vec<usize>)> {
    // Try to get from constant data first
    if let Some(tensor_data) = input.value() {
        // Try i64 first, then i32
        let indices: Vec<i64> = if let Ok(s) = tensor_data.as_slice::<i64>() {
            s.to_vec()
        } else if let Ok(s) = tensor_data.as_slice::<i32>() {
            s.iter().map(|&x| x as i64).collect()
        } else {
            return Err(anyhow!("Indices must be int32 or int64"));
        };

        let shape: Vec<usize> = tensor_data.shape.iter().map(|&d| d as usize).collect();
        return Ok((indices, shape));
    }

    // Try to get from value store (runtime indices)
    if let Some(dyn_tensor) = values.get(&input.name) {
        // For runtime indices, we need to extract the values
        // This is a simplified approach - in practice we'd need to handle this properly
        let tensor = dyn_tensor.as_rank4();
        let data = tensor.to_data();
        let floats = data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not get tensor data: {:?}", e))?;
        let indices: Vec<i64> = floats.iter().map(|&x| x as i64).collect();
        let shape = dyn_tensor.shape().to_vec();
        return Ok((indices, shape));
    }

    Err(anyhow!("Indices tensor '{}' not found", input.name))
}

/// Gather operator - gathers elements from input along axis using indices
/// output[i][j][k] = input[indices[i][j][k]][j][k] (for axis=0)
pub fn gather<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Gather(n) = node {
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;

        let data = get_input_tensor(&n.inputs[0], values, device)?;
        let (indices, indices_shape) = get_indices_tensor(&n.inputs[1], values, device)?;

        let data_tensor = data.as_rank4();
        let data_shape = data.shape().to_vec();
        let data_rank = data.rank();

        // For simple 2D gather (common case like embedding lookup)
        // data: [vocab_size, embed_dim], indices: [seq_len], axis=0
        // output: [seq_len, embed_dim]
        if data_rank == 2 && indices_shape.len() == 1 && axis == 0 {
            let vocab_size = data_shape[0];
            let embed_dim = data_shape[1];
            let seq_len = indices_shape[0];

            // Get data as 2D
            let data_2d: Tensor<B, 2> = data_tensor.reshape([vocab_size, embed_dim]);

            // Create output tensor by gathering rows
            let mut output_data = Vec::with_capacity(seq_len * embed_dim);
            let data_values = data_2d.to_data();
            let data_slice = data_values
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not get data slice: {:?}", e))?;

            for &idx in &indices {
                // Handle negative indices
                let actual_idx = if idx < 0 {
                    (vocab_size as i64 + idx) as usize
                } else {
                    idx as usize
                };

                if actual_idx >= vocab_size {
                    return Err(anyhow!(
                        "Index {} out of bounds for axis 0 with size {}",
                        idx,
                        vocab_size
                    ));
                }

                let start = actual_idx * embed_dim;
                let end = start + embed_dim;
                output_data.extend_from_slice(&data_slice[start..end]);
            }

            let output_tensor_data = TensorData::new(output_data, [seq_len, embed_dim]);
            let output: Tensor<B, 2> = Tensor::from_data(output_tensor_data, device);
            values.insert(output_name.clone(), DynTensor::from_rank2(output));
            return Ok(());
        }

        // For 3D data with 1D indices (batched embedding)
        if data_rank == 3 && indices_shape.len() == 1 && axis == 0 {
            let dim0 = data_shape[0];
            let dim1 = data_shape[1];
            let dim2 = data_shape[2];
            let seq_len = indices_shape[0];

            let data_3d: Tensor<B, 3> = data_tensor.reshape([dim0, dim1, dim2]);
            let data_values = data_3d.to_data();
            let data_slice = data_values
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not get data slice: {:?}", e))?;

            let mut output_data = Vec::with_capacity(seq_len * dim1 * dim2);
            let stride = dim1 * dim2;

            for &idx in &indices {
                let actual_idx = if idx < 0 {
                    (dim0 as i64 + idx) as usize
                } else {
                    idx as usize
                };

                if actual_idx >= dim0 {
                    return Err(anyhow!(
                        "Index {} out of bounds for axis 0 with size {}",
                        idx,
                        dim0
                    ));
                }

                let start = actual_idx * stride;
                let end = start + stride;
                output_data.extend_from_slice(&data_slice[start..end]);
            }

            let output_tensor_data = TensorData::new(output_data, [seq_len, dim1, dim2]);
            let output: Tensor<B, 3> = Tensor::from_data(output_tensor_data, device);
            values.insert(output_name.clone(), DynTensor::from_rank3(output));
            return Ok(());
        }

        // Generic fallback using rank-4 representation
        // This is less efficient but handles more cases
        let _dims = data_tensor.dims();
        let _normalized_axis = if axis < data_rank { axis } else { 0 };

        // For now, return error for unsupported configurations
        Err(anyhow!(
            "Gather: unsupported configuration - data_rank={}, indices_shape={:?}, axis={}. \
             Supported: 2D data with 1D indices (axis=0), 3D data with 1D indices (axis=0)",
            data_rank,
            indices_shape,
            axis
        ))
    } else {
        Err(anyhow!("Not a Gather node"))
    }
}

/// Where operator - element-wise ternary selection
/// output = condition ? x : y
pub fn where_op<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Where(n) = node {
        let output_name = &n.outputs[0].name;

        let condition = get_input_tensor(&n.inputs[0], values, device)?;
        let x = get_input_tensor(&n.inputs[1], values, device)?;
        let y = get_input_tensor(&n.inputs[2], values, device)?;

        // All tensors should broadcast to the same shape
        // Use rank-4 representation for broadcasting
        let cond_4d = condition.as_rank4();
        let x_4d = x.as_rank4();
        let y_4d = y.as_rank4();

        // Convert condition to bool mask (non-zero = true)
        // Using greater_elem with 0 as threshold
        let zero = Tensor::<B, 4>::zeros(cond_4d.dims(), device);
        let mask = cond_4d.greater(zero);

        // Select elements: where(mask, x, y) = mask * x + !mask * y
        let result = mask.clone().float() * x_4d.clone() + mask.bool_not().float() * y_4d;

        // Determine output rank
        let output_rank = condition.rank().max(x.rank()).max(y.rank());
        let output_dyn = match output_rank {
            1 => {
                let shape = result.dims();
                DynTensor::from_rank1(result.reshape([shape[3]]))
            }
            2 => {
                let shape = result.dims();
                DynTensor::from_rank2(result.reshape([shape[2], shape[3]]))
            }
            3 => {
                let shape = result.dims();
                DynTensor::from_rank3(result.reshape([shape[1], shape[2], shape[3]]))
            }
            4 => DynTensor::from_rank4(result),
            _ => return Err(anyhow!("Unsupported output rank for Where")),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Where node"))
    }
}

/// GatherElements operator - gathers elements based on indices (same rank as input)
pub fn gather_elements<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::GatherElements(n) = node {
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;

        let data = get_input_tensor(&n.inputs[0], values, device)?;
        let (indices, indices_shape) = get_indices_tensor(&n.inputs[1], values, device)?;

        let data_tensor = data.as_rank4();
        let data_shape = data.shape().to_vec();
        let data_rank = data.rank();

        // GatherElements requires indices and data to have the same rank
        if indices_shape.len() != data_rank {
            return Err(anyhow!(
                "GatherElements: indices rank ({}) must match data rank ({})",
                indices_shape.len(),
                data_rank
            ));
        }

        // Simple case: 2D data and indices
        if data_rank == 2 {
            let data_2d: Tensor<B, 2> = data_tensor.reshape([data_shape[0], data_shape[1]]);
            let data_values = data_2d.to_data();
            let data_slice = data_values
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not get data slice: {:?}", e))?;

            let mut output_data = Vec::with_capacity(indices.len());
            let rows = indices_shape[0];
            let cols = indices_shape[1];

            for i in 0..rows {
                for j in 0..cols {
                    let idx_pos = i * cols + j;
                    let idx = indices[idx_pos];

                    // Handle negative indices
                    let actual_idx = if idx < 0 {
                        (data_shape[axis] as i64 + idx) as usize
                    } else {
                        idx as usize
                    };

                    // Compute source position based on axis
                    let src_idx = if axis == 0 {
                        actual_idx * data_shape[1] + j
                    } else {
                        i * data_shape[1] + actual_idx
                    };

                    output_data.push(data_slice[src_idx]);
                }
            }

            let output_tensor_data = TensorData::new(output_data, [rows, cols]);
            let output: Tensor<B, 2> = Tensor::from_data(output_tensor_data, device);
            values.insert(output_name.clone(), DynTensor::from_rank2(output));
            return Ok(());
        }

        Err(anyhow!(
            "GatherElements: unsupported configuration - data_rank={}, axis={}",
            data_rank,
            axis
        ))
    } else {
        Err(anyhow!("Not a GatherElements node"))
    }
}

/// TopK operator - returns top k largest/smallest elements
pub fn topk<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::TopK(n) = node {
        let values_output_name = &n.outputs[0].name;
        let indices_output_name = &n.outputs[1].name;
        let axis = n.config.axis;

        let data = get_input_tensor(&n.inputs[0], values, device)?;
        let data_tensor = data.as_rank4();
        let data_shape = data.shape().to_vec();
        let data_rank = data.rank();

        // Get k value
        let k = match &n.config.k {
            onnx_ir::node::topk::TopKInput::Static(k) => *k,
            onnx_ir::node::topk::TopKInput::Runtime(_) => {
                // Try to get k from input
                if n.inputs.len() > 1 {
                    let (k_indices, _) = get_indices_tensor(&n.inputs[1], values, device)?;
                    k_indices[0] as usize
                } else {
                    return Err(anyhow!("TopK: k value not provided"));
                }
            }
        };

        // Simple case: 1D or last-axis TopK
        if data_rank == 1 || axis == data_rank - 1 {
            let flat_size: usize = data_shape.iter().product();
            let inner_size = data_shape.last().copied().unwrap_or(1);
            let outer_size = flat_size / inner_size;

            let data_values = data_tensor.to_data();
            let data_slice = data_values
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not get data slice: {:?}", e))?;

            let mut output_values = Vec::with_capacity(outer_size * k);
            let mut output_indices = Vec::with_capacity(outer_size * k);

            for outer in 0..outer_size {
                let start = outer * inner_size;
                let end = start + inner_size;
                let slice = &data_slice[start..end];

                // Create (value, index) pairs and sort
                let mut pairs: Vec<(f32, usize)> = slice
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect();

                // Sort by value descending (largest first)
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k
                for (val, idx) in pairs.into_iter().take(k) {
                    output_values.push(val);
                    output_indices.push(idx as f32); // Store as f32 for now
                }
            }

            // Create output shape
            let mut output_shape = data_shape.clone();
            if let Some(last) = output_shape.last_mut() {
                *last = k;
            }

            match data_rank {
                1 => {
                    let values_tensor: Tensor<B, 1> =
                        Tensor::from_data(TensorData::new(output_values, [k]), device);
                    let indices_tensor: Tensor<B, 1> =
                        Tensor::from_data(TensorData::new(output_indices, [k]), device);
                    values.insert(
                        values_output_name.clone(),
                        DynTensor::from_rank1(values_tensor),
                    );
                    values.insert(
                        indices_output_name.clone(),
                        DynTensor::from_rank1(indices_tensor),
                    );
                }
                2 => {
                    let values_tensor: Tensor<B, 2> = Tensor::from_data(
                        TensorData::new(output_values, [output_shape[0], k]),
                        device,
                    );
                    let indices_tensor: Tensor<B, 2> = Tensor::from_data(
                        TensorData::new(output_indices, [output_shape[0], k]),
                        device,
                    );
                    values.insert(
                        values_output_name.clone(),
                        DynTensor::from_rank2(values_tensor),
                    );
                    values.insert(
                        indices_output_name.clone(),
                        DynTensor::from_rank2(indices_tensor),
                    );
                }
                _ => {
                    return Err(anyhow!("TopK: unsupported rank {}", data_rank));
                }
            }

            return Ok(());
        }

        Err(anyhow!(
            "TopK: unsupported configuration - data_rank={}, axis={}",
            data_rank,
            axis
        ))
    } else {
        Err(anyhow!("Not a TopK node"))
    }
}

/// CumSum operator - cumulative sum along an axis
pub fn cumsum<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::CumSum(n) = node {
        let output_name = &n.outputs[0].name;

        let data = get_input_tensor(&n.inputs[0], values, device)?;
        let data_tensor = data.as_rank4();
        let data_shape = data.shape().to_vec();
        let data_rank = data.rank();

        // Get axis from second input or config
        let axis = if n.inputs.len() > 1 {
            let (axis_vals, _) = get_indices_tensor(&n.inputs[1], values, device)?;
            let mut a = axis_vals[0] as i64;
            if a < 0 {
                a += data_rank as i64;
            }
            a as usize
        } else {
            0 // Default axis
        };

        // Get exclusive and reverse attributes from config
        let exclusive = n.config.exclusive;
        let reverse = n.config.reverse;

        // Simple 1D case
        if data_rank == 1 {
            let data_values = data_tensor.to_data();
            let data_slice = data_values
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not get data slice: {:?}", e))?;

            let mut output = vec![0.0f32; data_slice.len()];
            let len = data_slice.len();

            if reverse {
                let mut sum = 0.0f32;
                for i in (0..len).rev() {
                    if exclusive {
                        output[i] = sum;
                        sum += data_slice[i];
                    } else {
                        sum += data_slice[i];
                        output[i] = sum;
                    }
                }
            } else {
                let mut sum = 0.0f32;
                for i in 0..len {
                    if exclusive {
                        output[i] = sum;
                        sum += data_slice[i];
                    } else {
                        sum += data_slice[i];
                        output[i] = sum;
                    }
                }
            }

            let output_tensor: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(output, [len]), device);
            values.insert(output_name.clone(), DynTensor::from_rank1(output_tensor));
            return Ok(());
        }

        // 2D case with axis
        if data_rank == 2 {
            let rows = data_shape[0];
            let cols = data_shape[1];
            let data_2d: Tensor<B, 2> = data_tensor.reshape([rows, cols]);
            let data_values = data_2d.to_data();
            let data_slice = data_values
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not get data slice: {:?}", e))?;

            let mut output = vec![0.0f32; rows * cols];

            if axis == 0 {
                // Cumsum along rows
                for j in 0..cols {
                    if reverse {
                        let mut sum = 0.0f32;
                        for i in (0..rows).rev() {
                            let idx = i * cols + j;
                            if exclusive {
                                output[idx] = sum;
                                sum += data_slice[idx];
                            } else {
                                sum += data_slice[idx];
                                output[idx] = sum;
                            }
                        }
                    } else {
                        let mut sum = 0.0f32;
                        for i in 0..rows {
                            let idx = i * cols + j;
                            if exclusive {
                                output[idx] = sum;
                                sum += data_slice[idx];
                            } else {
                                sum += data_slice[idx];
                                output[idx] = sum;
                            }
                        }
                    }
                }
            } else {
                // Cumsum along cols
                for i in 0..rows {
                    if reverse {
                        let mut sum = 0.0f32;
                        for j in (0..cols).rev() {
                            let idx = i * cols + j;
                            if exclusive {
                                output[idx] = sum;
                                sum += data_slice[idx];
                            } else {
                                sum += data_slice[idx];
                                output[idx] = sum;
                            }
                        }
                    } else {
                        let mut sum = 0.0f32;
                        for j in 0..cols {
                            let idx = i * cols + j;
                            if exclusive {
                                output[idx] = sum;
                                sum += data_slice[idx];
                            } else {
                                sum += data_slice[idx];
                                output[idx] = sum;
                            }
                        }
                    }
                }
            }

            let output_tensor: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(output, [rows, cols]), device);
            values.insert(output_name.clone(), DynTensor::from_rank2(output_tensor));
            return Ok(());
        }

        Err(anyhow!(
            "CumSum: unsupported configuration - data_rank={}, axis={}",
            data_rank,
            axis
        ))
    } else {
        Err(anyhow!("Not a CumSum node"))
    }
}

/// NonZero operator - returns indices of non-zero elements
/// Output shape: [input_rank, num_nonzero_elements]
pub fn nonzero<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::NonZero(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for NonZero", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let input_rank = input_shape.len();

        // Get input data as flat array
        let input_4d = input_dyn.as_rank4();
        let input_data = input_4d.to_data();
        let input_slice: Vec<f32> = input_data
            .to_vec()
            .map_err(|e| anyhow!("Could not get input data: {:?}", e))?;

        // Find all non-zero element indices
        // We need to track multi-dimensional indices
        let total_elements: usize = input_shape.iter().product();

        // Calculate strides for converting flat index to multi-dimensional index
        let mut strides = vec![1usize; input_rank];
        if input_rank > 0 {
            for i in (0..input_rank - 1).rev() {
                strides[i] = strides[i + 1] * input_shape[i + 1];
            }
        }

        // Collect indices of non-zero elements
        let mut nonzero_indices: Vec<Vec<i64>> = vec![Vec::new(); input_rank];

        for flat_idx in 0..total_elements {
            let value = input_slice[flat_idx];
            if value != 0.0 {
                // Convert flat index to multi-dimensional indices
                let mut remaining = flat_idx;
                for dim in 0..input_rank {
                    let coord = remaining / strides[dim];
                    remaining %= strides[dim];
                    nonzero_indices[dim].push(coord as i64);
                }
            }
        }

        let num_nonzero = if input_rank > 0 {
            nonzero_indices[0].len()
        } else {
            0
        };

        // Create output tensor of shape [input_rank, num_nonzero]
        // Stored as f32 since DynTensor is float-based
        let mut output_data = Vec::with_capacity(input_rank * num_nonzero);
        for dim in 0..input_rank {
            for &idx in &nonzero_indices[dim] {
                output_data.push(idx as f32);
            }
        }

        // Handle edge case of empty output
        if num_nonzero == 0 {
            // Return empty tensor with shape [input_rank, 0]
            // Since we can't have actual 0-size tensors easily, create minimal tensor
            let output_tensor: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(vec![0.0f32; input_rank], [input_rank, 1]),
                device,
            );
            // Note: This is a workaround - ideally we'd return [input_rank, 0] shape
            values.insert(output_name.clone(), DynTensor::from_rank2(output_tensor));
        } else {
            let output_tensor: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(output_data, [input_rank, num_nonzero]),
                device,
            );
            values.insert(output_name.clone(), DynTensor::from_rank2(output_tensor));
        }

        Ok(())
    } else {
        Err(anyhow!("Not a NonZero node"))
    }
}

/// OneHot operator - produces a one-hot encoded tensor from indices
/// Given indices of shape [a, b, c], depth d, and axis a, produces output of shape [a, d, b, c]
pub fn one_hot<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::OneHot(n) = node {
        let output_name = &n.outputs[0].name;

        // Get indices - can be from value store or constant
        let (indices, indices_shape) = get_indices_tensor(&n.inputs[0], values, device)?;
        let indices_rank = indices_shape.len();

        // Get depth
        use onnx_ir::node::one_hot::OneHotDepthInput;
        let depth = match &n.config.depth {
            OneHotDepthInput::Static(d) => *d,
            OneHotDepthInput::Runtime(_) => {
                // Try to get from input
                if n.inputs.len() > 1 {
                    if let Some(depth_data) = n.inputs[1].value() {
                        let depth_slice = depth_data
                            .as_slice::<i64>()
                            .map_err(|e| anyhow!("Could not convert depth to i64: {:?}", e))?;
                        depth_slice[0] as usize
                    } else {
                        return Err(anyhow!("OneHot: runtime depth not yet supported"));
                    }
                } else {
                    return Err(anyhow!("OneHot: depth not provided"));
                }
            }
        };

        // Get values (off_value, on_value)
        use onnx_ir::node::one_hot::OneHotValuesInput;
        let (off_value, on_value) = match &n.config.values {
            OneHotValuesInput::Static(vals) => (vals[0], vals[1]),
            OneHotValuesInput::Runtime(_) => {
                // Try to get from input
                if n.inputs.len() > 2 {
                    if let Some(values_data) = n.inputs[2].value() {
                        let values_slice = values_data
                            .as_slice::<f32>()
                            .map_err(|e| anyhow!("Could not convert values to f32: {:?}", e))?;
                        (values_slice[0], values_slice[1])
                    } else {
                        return Err(anyhow!("OneHot: runtime values not yet supported"));
                    }
                } else {
                    (0.0, 1.0) // Default values
                }
            }
        };

        // Get axis (defaults to -1 which means last axis of output)
        let axis = n.config.axis;

        // Output rank is indices_rank + 1
        let output_rank = indices_rank + 1;

        // Normalize axis to positive value
        let normalized_axis = if axis < 0 {
            (output_rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        // Calculate output shape
        // Insert depth dimension at the specified axis
        let mut output_shape: Vec<usize> = indices_shape.clone();
        output_shape.insert(normalized_axis, depth);

        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![off_value; output_size];

        // For each index, set the corresponding position to on_value
        // The output position depends on the axis

        // Calculate strides for output tensor
        let mut output_strides = vec![1usize; output_rank];
        for i in (0..output_rank - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        // Calculate strides for indices tensor (without the one-hot dimension)
        let mut indices_strides = vec![1usize; indices_rank];
        if indices_rank > 0 {
            for i in (0..indices_rank - 1).rev() {
                indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
            }
        }

        // Iterate over all indices positions
        let total_indices: usize = indices_shape.iter().product();
        for flat_idx in 0..total_indices {
            // Get the index value at this position
            let idx = indices[flat_idx];

            // Handle negative indices and out-of-bounds
            let actual_idx = if idx < 0 {
                // Negative indices are treated as -1 (off_value position)
                continue;
            } else if idx as usize >= depth {
                // Out of bounds - position gets off_value (already set)
                continue;
            } else {
                idx as usize
            };

            // Convert flat index to multi-dimensional indices
            let mut indices_coords = vec![0usize; indices_rank];
            let mut remaining = flat_idx;
            for i in 0..indices_rank {
                indices_coords[i] = remaining / indices_strides[i];
                remaining %= indices_strides[i];
            }

            // Build output coordinates by inserting the one-hot index at the axis position
            let mut output_coords = Vec::with_capacity(output_rank);
            let mut indices_dim = 0;
            for dim in 0..output_rank {
                if dim == normalized_axis {
                    output_coords.push(actual_idx);
                } else {
                    output_coords.push(indices_coords[indices_dim]);
                    indices_dim += 1;
                }
            }

            // Calculate flat output index
            let mut output_flat_idx = 0;
            for (dim, &coord) in output_coords.iter().enumerate() {
                output_flat_idx += coord * output_strides[dim];
            }

            output_data[output_flat_idx] = on_value;
        }

        // Create output tensor based on rank
        let output_dyn = match output_rank {
            1 => {
                let tensor: Tensor<B, 1> =
                    Tensor::from_data(TensorData::new(output_data, [output_shape[0]]), device);
                DynTensor::from_rank1(tensor)
            }
            2 => {
                let tensor: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(output_data, [output_shape[0], output_shape[1]]),
                    device,
                );
                DynTensor::from_rank2(tensor)
            }
            3 => {
                let tensor: Tensor<B, 3> = Tensor::from_data(
                    TensorData::new(
                        output_data,
                        [output_shape[0], output_shape[1], output_shape[2]],
                    ),
                    device,
                );
                DynTensor::from_rank3(tensor)
            }
            4 => {
                let tensor: Tensor<B, 4> = Tensor::from_data(
                    TensorData::new(
                        output_data,
                        [
                            output_shape[0],
                            output_shape[1],
                            output_shape[2],
                            output_shape[3],
                        ],
                    ),
                    device,
                );
                DynTensor::from_rank4(tensor)
            }
            _ => return Err(anyhow!("OneHot: unsupported output rank {}", output_rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a OneHot node"))
    }
}

/// ScatterND operator - scatters updates into a copy of data at indices
/// output = copy of data, with updates scattered at positions specified by indices
pub fn scatter_nd<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::ScatterND(n) = node {
        let output_name = &n.outputs[0].name;

        // Get data tensor
        let data_dyn = get_input_tensor(&n.inputs[0], values, device)?;
        let data_shape = data_dyn.shape().to_vec();
        let data_rank = data_shape.len();

        // Get indices tensor
        let (indices, indices_shape) = get_indices_tensor(&n.inputs[1], values, device)?;
        let indices_rank = indices_shape.len();

        // Get updates tensor
        let updates_dyn = get_input_tensor(&n.inputs[2], values, device)?;

        // Get data as flat array
        let data_4d = data_dyn.as_rank4();
        let data_tensor_data = data_4d.to_data();
        let mut output_data: Vec<f32> = data_tensor_data
            .to_vec()
            .map_err(|e| anyhow!("ScatterND: cannot get data: {:?}", e))?;

        // Get updates as flat array
        let updates_4d = updates_dyn.as_rank4();
        let updates_tensor_data = updates_4d.to_data();
        let updates_data: Vec<f32> = updates_tensor_data
            .to_vec()
            .map_err(|e| anyhow!("ScatterND: cannot get updates: {:?}", e))?;

        // indices has shape [..., k] where k is the number of dimensions to index into data
        // k = indices_shape[indices_rank - 1]
        let k = if indices_rank > 0 {
            indices_shape[indices_rank - 1]
        } else {
            1
        };

        // Calculate strides for data tensor
        let mut data_strides = vec![1usize; data_rank];
        for i in (0..data_rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        // Number of scatter operations = product of indices shape except last dim
        let num_scatters: usize = if indices_rank > 1 {
            indices_shape[..indices_rank - 1].iter().product()
        } else {
            1
        };

        // Size of each update slice = product of data shape from k onwards
        let update_slice_size: usize = if k < data_rank {
            data_shape[k..].iter().product()
        } else {
            1
        };

        // Perform scatter
        for scatter_idx in 0..num_scatters {
            // Get the k-dimensional index for this scatter
            let indices_offset = scatter_idx * k;
            let mut data_flat_idx = 0usize;

            for dim in 0..k {
                let idx = indices[indices_offset + dim];
                let actual_idx = if idx < 0 {
                    (data_shape[dim] as i64 + idx) as usize
                } else {
                    idx as usize
                };

                if actual_idx >= data_shape[dim] {
                    return Err(anyhow!(
                        "ScatterND: index {} out of bounds for dim {} with size {}",
                        idx,
                        dim,
                        data_shape[dim]
                    ));
                }

                data_flat_idx += actual_idx * data_strides[dim];
            }

            // Copy update slice to output
            let updates_offset = scatter_idx * update_slice_size;
            for i in 0..update_slice_size {
                output_data[data_flat_idx + i] = updates_data[updates_offset + i];
            }
        }

        // Create output tensor
        let output_dyn = match data_rank {
            1 => {
                let tensor: Tensor<B, 1> =
                    Tensor::from_data(TensorData::new(output_data, [data_shape[0]]), device);
                DynTensor::from_rank1(tensor)
            }
            2 => {
                let tensor: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(output_data, [data_shape[0], data_shape[1]]),
                    device,
                );
                DynTensor::from_rank2(tensor)
            }
            3 => {
                let tensor: Tensor<B, 3> = Tensor::from_data(
                    TensorData::new(output_data, [data_shape[0], data_shape[1], data_shape[2]]),
                    device,
                );
                DynTensor::from_rank3(tensor)
            }
            4 => {
                let tensor: Tensor<B, 4> = Tensor::from_data(
                    TensorData::new(
                        output_data,
                        [data_shape[0], data_shape[1], data_shape[2], data_shape[3]],
                    ),
                    device,
                );
                DynTensor::from_rank4(tensor)
            }
            _ => return Err(anyhow!("ScatterND: unsupported data rank {}", data_rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ScatterND node"))
    }
}
