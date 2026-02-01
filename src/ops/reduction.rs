use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use onnx_ir::ir::Node;

/// Helper to perform reduction operation with keepdims handling
fn reduce_with_keepdims<B: Backend, F>(
    input_dyn: &DynTensor<B>,
    dims: &[usize],
    keepdims: bool,
    reduce_fn: F,
) -> Result<DynTensor<B>>
where
    F: Fn(Tensor<B, 4>, usize) -> Tensor<B, 4>,
{
    let input_shape = input_dyn.shape().to_vec();
    let rank = input_shape.len();
    let input_4d = input_dyn.as_rank4();

    // Map dims to rank-4 dimensions
    let rank4_offset = 4 - rank;
    let mut rank4_dims: Vec<usize> = dims.iter().map(|&d| d + rank4_offset).collect();
    rank4_dims.sort();
    rank4_dims.reverse(); // Process from highest to lowest to avoid index shifting

    // Apply reduction along each dimension
    let mut result = input_4d;
    for &dim in &rank4_dims {
        result = reduce_fn(result, dim);
    }

    // Compute output shape
    if keepdims {
        let mut output_shape = input_shape.clone();
        for &d in dims {
            output_shape[d] = 1;
        }

        let output_dyn = match output_shape.len() {
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
            4 => {
                let output: Tensor<B, 4> = result.reshape([
                    output_shape[0],
                    output_shape[1],
                    output_shape[2],
                    output_shape[3],
                ]);
                DynTensor::from_rank4(output)
            }
            _ => return Err(anyhow!("Unsupported output rank {}", output_shape.len())),
        };
        Ok(output_dyn)
    } else {
        // Remove reduced dimensions
        let output_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &d)| d)
            .collect();

        let output_dyn = match output_shape.len() {
            0 => {
                // Scalar result - store as rank 1 tensor
                let output: Tensor<B, 1> = result.reshape([1]);
                DynTensor::from_rank1(output)
            }
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
            4 => {
                let output: Tensor<B, 4> = result.reshape([
                    output_shape[0],
                    output_shape[1],
                    output_shape[2],
                    output_shape[3],
                ]);
                DynTensor::from_rank4(output)
            }
            _ => return Err(anyhow!("Unsupported output rank {}", output_shape.len())),
        };
        Ok(output_dyn)
    }
}

pub fn reduce_sum<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::ReduceSum(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let dims = &n.config.dims;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ReduceSum", input_name))?;

        let dims_to_reduce: Vec<usize> = if dims.is_empty() {
            // Reduce all dimensions
            (0..input_dyn.rank()).collect()
        } else {
            dims.clone()
        };

        let output_dyn = reduce_with_keepdims(&input_dyn, &dims_to_reduce, keepdims, |t, dim| {
            t.sum_dim(dim)
        })?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ReduceSum node"))
    }
}

pub fn reduce_mean<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::ReduceMean(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let dims = &n.config.dims;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ReduceMean", input_name))?;

        let dims_to_reduce: Vec<usize> = if dims.is_empty() {
            (0..input_dyn.rank()).collect()
        } else {
            dims.clone()
        };

        let output_dyn = reduce_with_keepdims(&input_dyn, &dims_to_reduce, keepdims, |t, dim| {
            t.mean_dim(dim)
        })?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ReduceMean node"))
    }
}

pub fn reduce_max<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::ReduceMax(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let dims = &n.config.dims;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ReduceMax", input_name))?;

        let dims_to_reduce: Vec<usize> = if dims.is_empty() {
            (0..input_dyn.rank()).collect()
        } else {
            dims.clone()
        };

        let output_dyn = reduce_with_keepdims(&input_dyn, &dims_to_reduce, keepdims, |t, dim| {
            t.max_dim(dim)
        })?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ReduceMax node"))
    }
}

pub fn reduce_min<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::ReduceMin(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let dims = &n.config.dims;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ReduceMin", input_name))?;

        let dims_to_reduce: Vec<usize> = if dims.is_empty() {
            (0..input_dyn.rank()).collect()
        } else {
            dims.clone()
        };

        let output_dyn = reduce_with_keepdims(&input_dyn, &dims_to_reduce, keepdims, |t, dim| {
            t.min_dim(dim)
        })?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ReduceMin node"))
    }
}

pub fn reduce_prod<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::ReduceProd(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let dims = &n.config.dims;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ReduceProd", input_name))?;

        let dims_to_reduce: Vec<usize> = if dims.is_empty() {
            (0..input_dyn.rank()).collect()
        } else {
            dims.clone()
        };

        let output_dyn = reduce_with_keepdims(&input_dyn, &dims_to_reduce, keepdims, |t, dim| {
            t.prod_dim(dim)
        })?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ReduceProd node"))
    }
}

pub fn argmax<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::ArgMax(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ArgMax", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();
        let input_4d = input_dyn.as_rank4();

        // Map axis to rank-4 dimension
        let rank4_axis = axis + (4 - rank);

        // Argmax returns indices as integers
        let result = input_4d.argmax(rank4_axis);

        // Compute output shape
        let mut output_shape = input_shape.clone();
        if keepdims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }

        // Convert argmax result to tensor and reshape
        use burn::tensor::TensorData;
        let indices_data = result.to_data();
        let indices_vec: Vec<i64> = indices_data.to_vec().unwrap();

        let output_dyn = match output_shape.len() {
            0 => {
                let tensor_data = TensorData::new(indices_vec, vec![1]);
                let output: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank1(output)
            }
            1 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank1(output)
            }
            2 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 2> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank2(output)
            }
            3 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 3> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank3(output)
            }
            4 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 4> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank4(output)
            }
            _ => {
                return Err(anyhow!(
                    "ArgMax: unsupported output rank {}",
                    output_shape.len()
                ))
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an ArgMax node"))
    }
}

pub fn argmin<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::ArgMin(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;
        let keepdims = n.config.keepdims;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ArgMin", input_name))?;

        let input_shape = input_dyn.shape().to_vec();
        let rank = input_shape.len();
        let input_4d = input_dyn.as_rank4();

        // Map axis to rank-4 dimension
        let rank4_axis = axis + (4 - rank);

        // Argmin returns indices as integers
        let result = input_4d.argmin(rank4_axis);

        // Compute output shape
        let mut output_shape = input_shape.clone();
        if keepdims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }

        // Convert argmin result to tensor and reshape
        use burn::tensor::TensorData;
        let indices_data = result.to_data();
        let indices_vec: Vec<i64> = indices_data.to_vec().unwrap();

        let output_dyn = match output_shape.len() {
            0 => {
                let tensor_data = TensorData::new(indices_vec, vec![1]);
                let output: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank1(output)
            }
            1 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank1(output)
            }
            2 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 2> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank2(output)
            }
            3 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 3> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank3(output)
            }
            4 => {
                let tensor_data = TensorData::new(indices_vec, output_shape.clone());
                let output: Tensor<B, 4> = Tensor::from_data(tensor_data, device);
                DynTensor::from_rank4(output)
            }
            _ => {
                return Err(anyhow!(
                    "ArgMin: unsupported output rank {}",
                    output_shape.len()
                ))
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an ArgMin node"))
    }
}
