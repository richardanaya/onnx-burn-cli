use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;

pub fn flatten<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Flatten(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis as i64;

        // Get input tensor
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Flatten", input_name))?;

        let shape = input_dyn.shape();
        let rank = shape.len();

        // Handle negative axis
        let axis = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        // Flatten: combine dimensions [0..axis] into first dim, [axis..] into second dim
        let first_dim: usize = shape[..axis].iter().product::<usize>().max(1);
        let second_dim: usize = shape[axis..].iter().product::<usize>().max(1);

        let input_4d = input_dyn.as_rank4();
        let output: Tensor<B, 2> = input_4d.reshape([first_dim, second_dim]);

        values.insert(output_name.clone(), DynTensor::from_rank2(output));
        Ok(())
    } else {
        Err(anyhow!("Not a Flatten node"))
    }
}

pub fn reshape<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Reshape(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        // Get input tensor
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Reshape", input_name))?;

        // Get target shape from the second input (shape tensor)
        let shape_arg = &n.inputs[1];
        let target_shape: Vec<i64> = if let Some(shape_data) = shape_arg.value() {
            shape_data
                .as_slice::<i64>()
                .map_err(|e| anyhow!("Could not read shape tensor: {:?}", e))?
                .to_vec()
        } else {
            // Try to get from value store (dynamic shape)
            return Err(anyhow!("Dynamic reshape shapes not yet supported"));
        };

        let input_4d = input_dyn.as_rank4();
        let total_elements: usize = input_dyn.shape().iter().product();

        // Resolve -1 in target shape
        let mut resolved_shape: Vec<usize> = Vec::new();
        let mut neg_one_idx: Option<usize> = None;
        let mut known_product: usize = 1;

        for (i, &dim) in target_shape.iter().enumerate() {
            if dim == -1 {
                if neg_one_idx.is_some() {
                    return Err(anyhow!("Only one dimension can be -1 in reshape"));
                }
                neg_one_idx = Some(i);
                resolved_shape.push(0); // placeholder
            } else if dim == 0 {
                // 0 means keep original dimension
                if i < input_dyn.shape().len() {
                    let orig = input_dyn.shape()[i];
                    resolved_shape.push(orig);
                    known_product *= orig;
                } else {
                    resolved_shape.push(1);
                    known_product *= 1;
                }
            } else {
                resolved_shape.push(dim as usize);
                known_product *= dim as usize;
            }
        }

        // Resolve -1
        if let Some(idx) = neg_one_idx {
            resolved_shape[idx] = total_elements / known_product;
        }

        // Reshape based on output rank
        let output_dyn = match resolved_shape.len() {
            1 => {
                let output: Tensor<B, 1> = input_4d.reshape([resolved_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = input_4d.reshape([resolved_shape[0], resolved_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    input_4d.reshape([resolved_shape[0], resolved_shape[1], resolved_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => {
                let output: Tensor<B, 4> = input_4d.reshape([
                    resolved_shape[0],
                    resolved_shape[1],
                    resolved_shape[2],
                    resolved_shape[3],
                ]);
                DynTensor::from_rank4(output)
            }
            _ => {
                return Err(anyhow!(
                    "Reshape: unsupported output rank {}",
                    resolved_shape.len()
                ))
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Reshape node"))
    }
}

pub fn shape<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Shape(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        // Get input tensor
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Shape", input_name))?;

        let shape_dim: Vec<usize> = vec![input_dyn.shape().len()];
        let shape_data = TensorData::new(
            input_dyn.shape().iter().map(|&d| d as i64).collect(),
            shape_dim,
        );
        let shape_tensor: Tensor<B, 1> = Tensor::from_data(shape_data, device);

        // Store shape output
        values.insert(output_name.clone(), DynTensor::from_rank1(shape_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Shape node"))
    }
}

pub fn squeeze<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Squeeze(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Squeeze", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();

        // Get axes to squeeze from config
        use onnx_ir::node::squeeze::SqueezeInput;
        let axes: Vec<usize> = match &n.config.axes {
            Some(SqueezeInput::Static(axes)) => axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect(),
            Some(SqueezeInput::Runtime(_)) => {
                return Err(anyhow!("Runtime squeeze axes not yet supported"));
            }
            None => {
                // Squeeze all dimensions of size 1
                input_shape
                    .iter()
                    .enumerate()
                    .filter(|(_, dim)| **dim == 1)
                    .map(|(i, _)| i)
                    .collect()
            }
        };

        // Build new shape by removing squeezed dimensions
        let new_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &dim)| dim)
            .collect();

        let input_4d = input_dyn.as_rank4();
        let output_dyn = match new_shape.len() {
            0 => {
                // Scalar output
                let output: Tensor<B, 1> = input_4d.reshape([1]);
                DynTensor::from_rank1(output)
            }
            1 => {
                let output: Tensor<B, 1> = input_4d.reshape([new_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = input_4d.reshape([new_shape[0], new_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    input_4d.reshape([new_shape[0], new_shape[1], new_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => {
                let output: Tensor<B, 4> =
                    input_4d.reshape([new_shape[0], new_shape[1], new_shape[2], new_shape[3]]);
                DynTensor::from_rank4(output)
            }
            _ => {
                return Err(anyhow!(
                    "Squeeze: unsupported output rank {}",
                    new_shape.len()
                ))
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Squeeze node"))
    }
}

pub fn unsqueeze<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Unsqueeze(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Unsqueeze", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let input_rank = input_shape.len();

        // Get axes to unsqueeze from config
        use onnx_ir::node::unsqueeze::UnsqueezeConfig;
        let axes: Vec<i64> = match &n.config {
            UnsqueezeConfig::Static(axes) => axes.clone(),
            UnsqueezeConfig::Runtime(_) => {
                return Err(anyhow!("Runtime unsqueeze axes not yet supported"));
            }
        };

        let output_rank = input_rank + axes.len();

        // Normalize negative axes
        let mut normalized_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (output_rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        normalized_axes.sort();

        // Build new shape by inserting 1s at specified axes
        let mut new_shape = Vec::with_capacity(output_rank);
        let mut input_idx = 0;
        for i in 0..output_rank {
            if normalized_axes.contains(&i) {
                new_shape.push(1);
            } else {
                if input_idx < input_shape.len() {
                    new_shape.push(input_shape[input_idx]);
                    input_idx += 1;
                }
            }
        }

        let input_4d = input_dyn.as_rank4();
        let output_dyn = match new_shape.len() {
            1 => {
                let output: Tensor<B, 1> = input_4d.reshape([new_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = input_4d.reshape([new_shape[0], new_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    input_4d.reshape([new_shape[0], new_shape[1], new_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => {
                let output: Tensor<B, 4> =
                    input_4d.reshape([new_shape[0], new_shape[1], new_shape[2], new_shape[3]]);
                DynTensor::from_rank4(output)
            }
            _ => {
                return Err(anyhow!(
                    "Unsqueeze: unsupported output rank {}",
                    new_shape.len()
                ))
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an Unsqueeze node"))
    }
}

pub fn transpose<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Transpose(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let perm = &n.config.perm;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Transpose", input_name))?;

        let input_shape = input_dyn.shape();
        let rank = input_shape.len();

        // Build output shape based on permutation
        let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();

        // For rank 4, use burn's permute
        let input_4d = input_dyn.as_rank4();

        // Convert perm to array format needed by burn
        // We need to handle different ranks by mapping through rank-4
        let output_dyn = match rank {
            2 => {
                // For 2D, map [0,1] perm to [0,1,2,3] perm for rank-4
                // Original dims are at positions 2,3 in rank-4
                let p: [usize; 4] = [0, 1, 2 + perm[0] as usize - 0, 2 + perm[1] as usize - 0];
                let p_normalized: [usize; 4] = if perm == &[1, 0] {
                    [0, 1, 3, 2]
                } else {
                    [0, 1, 2, 3]
                };
                let transposed = input_4d.permute(p_normalized);
                let output: Tensor<B, 2> = transposed.reshape([output_shape[0], output_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                // For 3D tensor stored as [1, d0, d1, d2], adjust permutation
                let p: [usize; 4] = [
                    0,
                    1 + perm[0] as usize,
                    1 + perm[1] as usize,
                    1 + perm[2] as usize,
                ];
                let transposed = input_4d.permute(p);
                let output: Tensor<B, 3> =
                    transposed.reshape([output_shape[0], output_shape[1], output_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => {
                let p: [usize; 4] = [
                    perm[0] as usize,
                    perm[1] as usize,
                    perm[2] as usize,
                    perm[3] as usize,
                ];
                let transposed = input_4d.permute(p);
                DynTensor::from_rank4(transposed)
            }
            _ => return Err(anyhow!("Transpose: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Transpose node"))
    }
}

pub fn concat<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Concat(n) = node {
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;

        // Collect all input tensors
        let mut input_tensors: Vec<Tensor<B, 4>> = Vec::new();
        for input in &n.inputs {
            let input_dyn = values
                .get(&input.name)
                .ok_or_else(|| anyhow!("Input tensor '{}' not found for Concat", input.name))?;
            input_tensors.push(input_dyn.as_rank4());
        }

        if input_tensors.is_empty() {
            return Err(anyhow!("Concat requires at least one input"));
        }

        // Get the first input's shape to determine output rank
        let first_input = values.get(&n.inputs[0].name).unwrap();
        let rank = first_input.rank();

        // Map axis to rank-4 dimension
        let rank4_axis = axis + (4 - rank);

        // Concatenate along the adjusted axis
        let output_4d = Tensor::cat(input_tensors, rank4_axis);

        // Compute output shape
        let mut output_shape = first_input.shape().to_vec();
        let mut concat_dim_size = 0;
        for input in &n.inputs {
            let input_dyn = values.get(&input.name).unwrap();
            concat_dim_size += input_dyn.shape()[axis];
        }
        output_shape[axis] = concat_dim_size;

        let output_dyn = match rank {
            1 => {
                let output: Tensor<B, 1> = output_4d.reshape([output_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = output_4d.reshape([output_shape[0], output_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    output_4d.reshape([output_shape[0], output_shape[1], output_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(output_4d),
            _ => return Err(anyhow!("Concat: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Concat node"))
    }
}

pub fn size<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Size(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Size", input_name))?;

        // Calculate total number of elements
        let total_elements: i64 = input_dyn.shape().iter().map(|&d| d as i64).product();

        // Create scalar tensor with the size
        let size_data = TensorData::new(vec![total_elements], vec![1]);
        let size_tensor: Tensor<B, 1> = Tensor::from_data(size_data, device);

        values.insert(output_name.clone(), DynTensor::from_rank1(size_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Size node"))
    }
}
