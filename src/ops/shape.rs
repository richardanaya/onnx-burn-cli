use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
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

pub fn split<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Split(n) = node {
        let input_name = &n.inputs[0].name;
        let axis = n.config.axis;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Split", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();
        let dim_size = input_shape[axis];

        // Determine split sizes
        let split_sizes: Vec<usize> = if let Some(ref split_input) = n.config.split_sizes {
            use onnx_ir::node::split::SplitSizesInput;
            match split_input {
                SplitSizesInput::Static(sizes) => sizes.clone(),
                SplitSizesInput::Runtime(_) => {
                    return Err(anyhow!("Runtime split sizes not yet supported"));
                }
            }
        } else if let Some(split_size) = n.config.split_size {
            // Uniform split
            let num_outputs = n.outputs.len();
            let mut sizes = vec![split_size; num_outputs];
            // Handle remainder for last split
            let total: usize = sizes.iter().sum();
            if total > dim_size {
                sizes[num_outputs - 1] = dim_size - (split_size * (num_outputs - 1));
            }
            sizes
        } else if let Some(num_outputs) = n.config.num_outputs {
            // Calculate from num_outputs
            let base_size = dim_size / num_outputs;
            let remainder = dim_size % num_outputs;
            let mut sizes = vec![base_size; num_outputs];
            // Distribute remainder
            for i in 0..remainder {
                sizes[i] += 1;
            }
            sizes
        } else {
            // Default: split evenly based on number of outputs
            let num_outputs = n.outputs.len();
            let base_size = dim_size / num_outputs;
            let remainder = dim_size % num_outputs;
            let mut sizes = vec![base_size; num_outputs];
            for i in 0..remainder {
                sizes[i] += 1;
            }
            sizes
        };

        // Get input as rank-4
        let input_4d = input_dyn.as_rank4();

        // Map axis to rank-4 dimension
        let rank4_axis = axis + (4 - rank);

        // Perform splits using narrow
        let mut offset = 0;
        for (i, &split_size) in split_sizes.iter().enumerate() {
            let output_name = &n.outputs[i].name;

            // Use narrow to extract slice along axis
            let sliced = input_4d.clone().narrow(rank4_axis, offset, split_size);
            offset += split_size;

            // Build output shape
            let mut output_shape = input_shape.clone();
            output_shape[axis] = split_size;

            let output_dyn = match rank {
                1 => {
                    let output: Tensor<B, 1> = sliced.reshape([output_shape[0]]);
                    DynTensor::from_rank1(output)
                }
                2 => {
                    let output: Tensor<B, 2> = sliced.reshape([output_shape[0], output_shape[1]]);
                    DynTensor::from_rank2(output)
                }
                3 => {
                    let output: Tensor<B, 3> =
                        sliced.reshape([output_shape[0], output_shape[1], output_shape[2]]);
                    DynTensor::from_rank3(output)
                }
                4 => DynTensor::from_rank4(sliced),
                _ => return Err(anyhow!("Split: unsupported rank {}", rank)),
            };

            values.insert(output_name.clone(), output_dyn);
        }

        Ok(())
    } else {
        Err(anyhow!("Not a Split node"))
    }
}

pub fn slice<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Slice(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Slice", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();

        // Extract slice parameters
        use onnx_ir::node::slice::SliceInput;

        let starts: Vec<i64> = match &n.config.starts {
            SliceInput::Static(s) => s.clone(),
            SliceInput::Runtime(_) => {
                return Err(anyhow!("Runtime slice starts not yet supported"))
            }
        };

        let ends: Vec<i64> = match &n.config.ends {
            SliceInput::Static(e) => e.clone(),
            SliceInput::Runtime(_) => return Err(anyhow!("Runtime slice ends not yet supported")),
        };

        let axes: Vec<i64> = match &n.config.axes {
            Some(SliceInput::Static(a)) => a.clone(),
            Some(SliceInput::Runtime(_)) => {
                return Err(anyhow!("Runtime slice axes not yet supported"))
            }
            None => (0..starts.len() as i64).collect(),
        };

        let steps: Vec<i64> = match &n.config.steps {
            Some(SliceInput::Static(s)) => s.clone(),
            Some(SliceInput::Runtime(_)) => {
                return Err(anyhow!("Runtime slice steps not yet supported"))
            }
            None => vec![1; starts.len()],
        };

        // Normalize and apply slices
        let input_4d = input_dyn.as_rank4();

        // Build ranges for each dimension
        let mut ranges: Vec<(usize, usize)> = input_shape.iter().map(|&d| (0, d)).collect();

        for i in 0..starts.len() {
            let axis = if axes[i] < 0 {
                (rank as i64 + axes[i]) as usize
            } else {
                axes[i] as usize
            };

            let dim_size = input_shape[axis] as i64;
            let step = steps[i];

            if step != 1 {
                return Err(anyhow!("Slice: step != 1 not yet supported"));
            }

            // Normalize start
            let mut start = starts[i];
            if start < 0 {
                start = (dim_size + start).max(0);
            }
            start = start.min(dim_size);

            // Normalize end
            let mut end = ends[i];
            if end < 0 {
                end = (dim_size + end).max(0);
            } else if end > dim_size {
                end = dim_size;
            }
            end = end.max(start);

            ranges[axis] = (start as usize, end as usize);
        }

        // Apply slices using narrow for each dimension
        let mut result = input_4d;
        for (axis, &(start, end)) in ranges.iter().enumerate() {
            let rank4_axis = axis + (4 - rank);
            let length = end - start;
            if length > 0 && (start > 0 || length < input_shape[axis]) {
                result = result.narrow(rank4_axis, start, length);
            }
        }

        // Calculate output shape
        let output_shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();

        let output_dyn = match rank {
            1 => {
                let output: Tensor<B, 1> = result.reshape([output_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = result.reshape([output_shape[0], output_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    result.reshape([output_shape[0], output_shape[1], output_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(result),
            _ => return Err(anyhow!("Slice: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Slice node"))
    }
}

pub fn expand<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Expand(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Expand", input_name))?;

        let input_shape = input_dyn.shape().to_vec();

        // Get target shape from config
        use onnx_ir::node::expand::ExpandConfig;

        let target_shape: Vec<i64> = match &n.config {
            ExpandConfig::Static(shape) => shape.clone(),
            ExpandConfig::Runtime(_) => {
                return Err(anyhow!("Runtime expand shape not yet supported"))
            }
        };

        // Resolve -1 values (copy from input)
        let mut resolved_shape: Vec<usize> = Vec::new();
        let input_rank = input_shape.len();
        let target_rank = target_shape.len();

        // Align shapes from the right
        for i in 0..target_rank {
            let target_dim = target_shape[i];
            let input_idx = if i >= target_rank - input_rank {
                Some(i - (target_rank - input_rank))
            } else {
                None
            };

            let dim = if target_dim == -1 || target_dim == 0 {
                // Copy from input if available
                if let Some(idx) = input_idx {
                    input_shape[idx]
                } else {
                    return Err(anyhow!("Expand: cannot infer dimension {} from input", i));
                }
            } else {
                target_dim as usize
            };

            resolved_shape.push(dim);
        }

        // Use burn's expand functionality via broadcast
        let input_4d = input_dyn.as_rank4();

        // Pad input shape to match target rank by prepending 1s
        let mut padded_input_shape = vec![1usize; target_rank.saturating_sub(input_rank)];
        padded_input_shape.extend(&input_shape);

        // Build the broadcast shape for rank-4
        let output_rank = resolved_shape.len();
        let mut broadcast_shape = [1usize; 4];
        for (i, &dim) in resolved_shape.iter().enumerate() {
            let idx = i + (4 - output_rank);
            broadcast_shape[idx] = dim;
        }

        // Use expand (broadcast) to reach target shape
        let expanded = input_4d.expand(broadcast_shape);

        let output_dyn = match output_rank {
            1 => {
                let output: Tensor<B, 1> = expanded.reshape([resolved_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = expanded.reshape([resolved_shape[0], resolved_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    expanded.reshape([resolved_shape[0], resolved_shape[1], resolved_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(expanded),
            _ => return Err(anyhow!("Expand: unsupported output rank {}", output_rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an Expand node"))
    }
}

pub fn tile<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Tile(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Tile", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();

        // Get repeats from config
        use onnx_ir::node::tile::TileInput;

        let repeats: Vec<usize> = match &n.config.repeats {
            TileInput::Static(r) => r.clone(),
            TileInput::Runtime(_) => return Err(anyhow!("Runtime tile repeats not yet supported")),
        };

        if repeats.len() != rank {
            return Err(anyhow!(
                "Tile: repeats length {} must match input rank {}",
                repeats.len(),
                rank
            ));
        }

        let input_4d = input_dyn.as_rank4();

        // Build output shape
        let output_shape: Vec<usize> = input_shape
            .iter()
            .zip(repeats.iter())
            .map(|(&dim, &rep)| dim * rep)
            .collect();

        // Use burn's repeat functionality
        // We need to repeat along each dimension
        let mut result = input_4d;
        for (i, &rep) in repeats.iter().enumerate() {
            if rep > 1 {
                let rank4_axis = i + (4 - rank);
                result = result.repeat_dim(rank4_axis, rep);
            }
        }

        let output_dyn = match rank {
            1 => {
                let output: Tensor<B, 1> = result.reshape([output_shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = result.reshape([output_shape[0], output_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    result.reshape([output_shape[0], output_shape[1], output_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(result),
            _ => return Err(anyhow!("Tile: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Tile node"))
    }
}

pub fn pad<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Pad(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Pad", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();

        // Get pad values from config
        use onnx_ir::node::pad::{ConstantValueInput, PadInput, PadMode};

        let pads: Vec<usize> = match &n.config.pads {
            PadInput::Static(p) => p.clone(),
            PadInput::Runtime(_) => return Err(anyhow!("Runtime pad values not yet supported")),
        };

        let constant_value: f32 = match &n.config.constant_value {
            ConstantValueInput::Static(v) => *v,
            ConstantValueInput::Runtime(_) => {
                return Err(anyhow!("Runtime constant value not yet supported"))
            }
        };

        // The pads from onnx-ir are reordered to [left, right, top, bottom]
        // for the last two dimensions only
        if pads.len() != 4 {
            return Err(anyhow!(
                "Pad: expected 4 pad values (left, right, top, bottom), got {}",
                pads.len()
            ));
        }

        let left = pads[0];
        let right = pads[1];
        let top = pads[2];
        let bottom = pads[3];

        // Get input as rank 4
        let input_4d = input_dyn.as_rank4();

        // Apply padding based on mode
        let padded = match n.config.mode {
            PadMode::Constant => {
                // Create pad config for burn
                // burn's pad_with_zeros pads last two dimensions
                // We need to use a more manual approach for constant padding
                let [b, c, h, w] = input_4d.dims();
                let new_h = h + top + bottom;
                let new_w = w + left + right;

                // Create output tensor filled with constant value
                let output_shape = [b, c, new_h, new_w];
                let mut output: Tensor<B, 4> = Tensor::full(output_shape, constant_value, device);

                // Copy input into the correct position
                // Using narrow and slice_assign
                let ranges = [0..b, 0..c, top..(top + h), left..(left + w)];
                output = output.slice_assign(ranges, input_4d);
                output
            }
            PadMode::Reflect => {
                return Err(anyhow!("Pad: reflect mode not yet implemented"));
            }
            PadMode::Edge => {
                return Err(anyhow!("Pad: edge mode not yet implemented"));
            }
        };

        // Calculate output shape
        let mut output_shape = input_shape.clone();
        output_shape[rank - 2] += top + bottom;
        output_shape[rank - 1] += left + right;

        let output_dyn = match rank {
            2 => {
                let output: Tensor<B, 2> = padded.reshape([output_shape[0], output_shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> =
                    padded.reshape([output_shape[0], output_shape[1], output_shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(padded),
            _ => return Err(anyhow!("Pad: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Pad node"))
    }
}

pub fn resize<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Resize(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Resize", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();

        if rank != 4 {
            return Err(anyhow!(
                "Resize: only 4D tensors (N,C,H,W) are supported, got rank {}",
                rank
            ));
        }

        let input_4d = input_dyn.as_rank4();

        // Determine the interpolation mode
        use onnx_ir::node::resize::ResizeMode;
        let mode = match &n.config.mode {
            ResizeMode::Nearest => InterpolateMode::Nearest,
            ResizeMode::Linear => InterpolateMode::Bilinear,
            ResizeMode::Cubic => InterpolateMode::Bicubic,
        };

        // Determine output size
        use onnx_ir::node::resize::{ResizeScales, ResizeSizes};

        let output_size: [usize; 2] = if let Some(ref sizes) = n.config.sizes {
            match sizes {
                ResizeSizes::Static(s) => {
                    // s contains only spatial dimensions (H, W) as extracted by onnx-ir
                    if s.len() >= 2 {
                        [s[0], s[1]]
                    } else if s.len() == 1 {
                        [s[0], s[0]]
                    } else {
                        return Err(anyhow!("Resize: invalid sizes"));
                    }
                }
                ResizeSizes::Runtime(_) => {
                    return Err(anyhow!("Resize: runtime sizes not yet supported"));
                }
            }
        } else if let Some(ref scales) = n.config.scales {
            match scales {
                ResizeScales::Static(s) => {
                    // s contains only spatial dimensions (H, W) as extracted by onnx-ir
                    let h = input_shape[2];
                    let w = input_shape[3];
                    if s.len() >= 2 {
                        [(h as f32 * s[0]) as usize, (w as f32 * s[1]) as usize]
                    } else if s.len() == 1 {
                        let scale = s[0];
                        [(h as f32 * scale) as usize, (w as f32 * scale) as usize]
                    } else {
                        return Err(anyhow!("Resize: invalid scales"));
                    }
                }
                ResizeScales::Runtime(_) => {
                    return Err(anyhow!("Resize: runtime scales not yet supported"));
                }
            }
        } else {
            return Err(anyhow!("Resize: either scales or sizes must be provided"));
        };

        // Apply interpolation
        let options = InterpolateOptions { mode };
        let output_4d = interpolate(input_4d, output_size, options);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_4d));
        Ok(())
    } else {
        Err(anyhow!("Not a Resize node"))
    }
}

pub fn depth_to_space<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::DepthToSpace(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let block_size = n.config.block_size;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for DepthToSpace", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        if input_shape.len() != 4 {
            return Err(anyhow!(
                "DepthToSpace: only rank 4 tensors supported, got rank {}",
                input_shape.len()
            ));
        }

        let [b, c, h, w] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        let bs2 = block_size * block_size;

        if c % bs2 != 0 {
            return Err(anyhow!(
                "DepthToSpace: channels {} must be divisible by blocksize^2 ({})",
                c,
                bs2
            ));
        }

        let new_c = c / bs2;
        let new_h = h * block_size;
        let new_w = w * block_size;

        let input_4d = input_dyn.as_rank4();

        // DepthToSpace transformation:
        // 1. Reshape from [N, C, H, W] to [N, bs, bs, C/(bs*bs), H, W] (DCR mode)
        //    or [N, C/(bs*bs), bs, bs, H, W] (CRD mode)
        // 2. Permute dimensions appropriately
        // 3. Reshape to [N, C/(bs*bs), H*bs, W*bs]

        use onnx_ir::node::depth_to_space::DepthToSpaceMode;

        // We implement this using reshape and permute operations
        // For DCR mode: reshape -> permute -> reshape
        // DCR: depth, column, row order
        // CRD: column, row, depth order

        let output = match n.config.mode {
            DepthToSpaceMode::Dcr => {
                // DCR mode (default):
                // Reshape to [N, bs, bs, C', H, W]
                // Permute to [N, C', H, bs, W, bs]
                // Reshape to [N, C', H*bs, W*bs]

                // Since burn tensors are fixed-rank, we'll use a different approach:
                // Flatten, rearrange indices, and reshape
                // This is mathematically equivalent

                // Step 1: Reshape to [N, bs, bs, new_c, H, W] by treating as [N*bs*bs*new_c, H, W]
                // and then reorganizing

                // Alternative: use slice and cat operations to rearrange
                // For simplicity, we'll use the direct mathematical approach

                // Create output tensor by copying values in the right order
                let input_data = input_4d.to_data();
                let input_slice: Vec<f32> = input_data.to_vec().unwrap();

                let mut output_data = vec![0.0f32; b * new_c * new_h * new_w];

                for n_idx in 0..b {
                    for c_idx in 0..new_c {
                        for h_idx in 0..new_h {
                            for w_idx in 0..new_w {
                                // Map output indices to input indices
                                let out_h_block = h_idx / block_size;
                                let out_h_offset = h_idx % block_size;
                                let out_w_block = w_idx / block_size;
                                let out_w_offset = w_idx % block_size;

                                // DCR: channel index = c_idx * bs^2 + out_h_offset * bs + out_w_offset
                                let in_c = c_idx * bs2 + out_h_offset * block_size + out_w_offset;
                                let in_h = out_h_block;
                                let in_w = out_w_block;

                                let in_idx = n_idx * c * h * w + in_c * h * w + in_h * w + in_w;
                                let out_idx = n_idx * new_c * new_h * new_w
                                    + c_idx * new_h * new_w
                                    + h_idx * new_w
                                    + w_idx;

                                output_data[out_idx] = input_slice[in_idx];
                            }
                        }
                    }
                }

                let output_tensor_data = TensorData::new(output_data, [b, new_c, new_h, new_w]);
                Tensor::from_data(output_tensor_data, &input_4d.device())
            }
            DepthToSpaceMode::Crd => {
                // CRD mode:
                // Different ordering of the depth dimension

                let input_data = input_4d.to_data();
                let input_slice: Vec<f32> = input_data.to_vec().unwrap();

                let mut output_data = vec![0.0f32; b * new_c * new_h * new_w];

                for n_idx in 0..b {
                    for c_idx in 0..new_c {
                        for h_idx in 0..new_h {
                            for w_idx in 0..new_w {
                                let out_h_block = h_idx / block_size;
                                let out_h_offset = h_idx % block_size;
                                let out_w_block = w_idx / block_size;
                                let out_w_offset = w_idx % block_size;

                                // CRD: channel index = (c_idx * bs + out_h_offset) * bs + out_w_offset
                                let in_c =
                                    (c_idx * block_size + out_h_offset) * block_size + out_w_offset;
                                let in_h = out_h_block;
                                let in_w = out_w_block;

                                let in_idx = n_idx * c * h * w + in_c * h * w + in_h * w + in_w;
                                let out_idx = n_idx * new_c * new_h * new_w
                                    + c_idx * new_h * new_w
                                    + h_idx * new_w
                                    + w_idx;

                                output_data[out_idx] = input_slice[in_idx];
                            }
                        }
                    }
                }

                let output_tensor_data = TensorData::new(output_data, [b, new_c, new_h, new_w]);
                Tensor::from_data(output_tensor_data, &input_4d.device())
            }
        };

        values.insert(output_name.clone(), DynTensor::from_rank4(output));
        Ok(())
    } else {
        Err(anyhow!("Not a DepthToSpace node"))
    }
}

pub fn space_to_depth<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::SpaceToDepth(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let block_size = n.config.block_size;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for SpaceToDepth", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        if input_shape.len() != 4 {
            return Err(anyhow!(
                "SpaceToDepth: only rank 4 tensors supported, got rank {}",
                input_shape.len()
            ));
        }

        let [b, c, h, w] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];

        if h % block_size != 0 || w % block_size != 0 {
            return Err(anyhow!(
                "SpaceToDepth: H ({}) and W ({}) must be divisible by blocksize ({})",
                h,
                w,
                block_size
            ));
        }

        let bs2 = block_size * block_size;
        let new_c = c * bs2;
        let new_h = h / block_size;
        let new_w = w / block_size;

        let input_4d = input_dyn.as_rank4();

        // SpaceToDepth is the inverse of DepthToSpace
        // Rearrange spatial blocks into the depth dimension

        let input_data = input_4d.to_data();
        let input_slice: Vec<f32> = input_data.to_vec().unwrap();

        let mut output_data = vec![0.0f32; b * new_c * new_h * new_w];

        for n_idx in 0..b {
            for c_idx in 0..new_c {
                for h_idx in 0..new_h {
                    for w_idx in 0..new_w {
                        // Map output indices to input indices
                        // Reverse of DCR mode DepthToSpace
                        let orig_c = c_idx / bs2;
                        let block_offset = c_idx % bs2;
                        let h_offset = block_offset / block_size;
                        let w_offset = block_offset % block_size;

                        let in_h = h_idx * block_size + h_offset;
                        let in_w = w_idx * block_size + w_offset;

                        let in_idx = n_idx * c * h * w + orig_c * h * w + in_h * w + in_w;
                        let out_idx = n_idx * new_c * new_h * new_w
                            + c_idx * new_h * new_w
                            + h_idx * new_w
                            + w_idx;

                        output_data[out_idx] = input_slice[in_idx];
                    }
                }
            }
        }

        let output_tensor_data = TensorData::new(output_data, [b, new_c, new_h, new_w]);
        let output: Tensor<B, 4> = Tensor::from_data(output_tensor_data, &input_4d.device());

        values.insert(output_name.clone(), DynTensor::from_rank4(output));
        Ok(())
    } else {
        Err(anyhow!("Not a SpaceToDepth node"))
    }
}
