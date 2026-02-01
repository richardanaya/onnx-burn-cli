use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;

/// Helper to get input tensor, either from value store or from constant data
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

        // Create tensor based on rank
        match shape.len() {
            0 => {
                // Scalar
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

pub fn add<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Add(n) = node {
        let output_name = &n.outputs[0].name;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        // For element-wise add, both tensors work on rank-4 internal representation
        // Broadcasting is handled automatically
        let a_tensor = a.as_rank4();
        let b_tensor = b.as_rank4();

        let result = a_tensor.add(b_tensor);

        // Output rank should match the higher rank input (after broadcasting)
        let output_rank = a.rank().max(b.rank());
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
            _ => return Err(anyhow!("Unsupported output rank")),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an Add node"))
    }
}

pub fn sub<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Sub(n) = node {
        let output_name = &n.outputs[0].name;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        let a_tensor = a.as_rank4();
        let b_tensor = b.as_rank4();

        let result = a_tensor.sub(b_tensor);

        let output_rank = a.rank().max(b.rank());
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
            _ => return Err(anyhow!("Unsupported output rank")),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Sub node"))
    }
}

pub fn mul<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Mul(n) = node {
        let output_name = &n.outputs[0].name;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        let a_tensor = a.as_rank4();
        let b_tensor = b.as_rank4();

        let result = a_tensor.mul(b_tensor);

        let output_rank = a.rank().max(b.rank());
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
            _ => return Err(anyhow!("Unsupported output rank")),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Mul node"))
    }
}

pub fn div<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Div(n) = node {
        let output_name = &n.outputs[0].name;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        let a_tensor = a.as_rank4();
        let b_tensor = b.as_rank4();

        let result = a_tensor.div(b_tensor);

        let output_rank = a.rank().max(b.rank());
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
            _ => return Err(anyhow!("Unsupported output rank")),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Div node"))
    }
}

/// Helper to create DynTensor from rank4 result with correct output rank
fn rank4_to_dyn_tensor<B: Backend>(
    result: Tensor<B, 4>,
    output_rank: usize,
) -> Result<DynTensor<B>> {
    let shape = result.dims();
    match output_rank {
        1 => Ok(DynTensor::from_rank1(result.reshape([shape[3]]))),
        2 => Ok(DynTensor::from_rank2(result.reshape([shape[2], shape[3]]))),
        3 => Ok(DynTensor::from_rank3(
            result.reshape([shape[1], shape[2], shape[3]]),
        )),
        4 => Ok(DynTensor::from_rank4(result)),
        _ => Err(anyhow!("Unsupported output rank: {}", output_rank)),
    }
}

pub fn pow<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Pow(n) = node {
        let output_name = &n.outputs[0].name;

        let base = get_input_tensor(&n.inputs[0], values, device)?;
        let exponent = get_input_tensor(&n.inputs[1], values, device)?;

        let base_tensor = base.as_rank4();
        let exp_tensor = exponent.as_rank4();

        // Element-wise power: base^exponent
        let result = base_tensor.powf(exp_tensor);

        let output_rank = base.rank().max(exponent.rank());
        let output_dyn = rank4_to_dyn_tensor(result, output_rank)?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Pow node"))
    }
}

pub fn max_elementwise<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Max(n) = node {
        let output_name = &n.outputs[0].name;

        // Max can have multiple inputs - accumulate element-wise max
        if n.inputs.is_empty() {
            return Err(anyhow!("Max requires at least one input"));
        }

        let first = get_input_tensor(&n.inputs[0], values, device)?;
        let mut result = first.as_rank4();
        let mut max_rank = first.rank();

        for input in n.inputs.iter().skip(1) {
            let tensor = get_input_tensor(input, values, device)?;
            max_rank = max_rank.max(tensor.rank());
            let t = tensor.as_rank4();
            // Element-wise max using mask: result = where(result > t, result, t)
            let mask = result.clone().greater(t.clone());
            result = mask.clone().float() * result + (mask.bool_not().float()) * t;
        }

        let output_dyn = rank4_to_dyn_tensor(result, max_rank)?;
        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Max node"))
    }
}

pub fn min_elementwise<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Min(n) = node {
        let output_name = &n.outputs[0].name;

        // Min can have multiple inputs - accumulate element-wise min
        if n.inputs.is_empty() {
            return Err(anyhow!("Min requires at least one input"));
        }

        let first = get_input_tensor(&n.inputs[0], values, device)?;
        let mut result = first.as_rank4();
        let mut max_rank = first.rank();

        for input in n.inputs.iter().skip(1) {
            let tensor = get_input_tensor(input, values, device)?;
            max_rank = max_rank.max(tensor.rank());
            let t = tensor.as_rank4();
            // Element-wise min using mask: result = where(result < t, result, t)
            let mask = result.clone().lower(t.clone());
            result = mask.clone().float() * result + (mask.bool_not().float()) * t;
        }

        let output_dyn = rank4_to_dyn_tensor(result, max_rank)?;
        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Min node"))
    }
}

pub fn modulo<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Mod(n) = node {
        let output_name = &n.outputs[0].name;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        let a_tensor = a.as_rank4();
        let b_tensor = b.as_rank4();

        // Modulo: a % b
        // If fmod is true, use floating-point mod (sign follows dividend)
        // If fmod is false (default), use integer mod (sign follows divisor)
        // For simplicity in floating-point, we implement: a - floor(a/b) * b
        // which matches Python's % behavior (sign follows divisor)
        let result = if n.config.fmod {
            // C-style fmod: a - trunc(a/b) * b (sign follows dividend)
            let div_result = a_tensor.clone().div(b_tensor.clone());
            // trunc = floor for positive, ceil for negative
            // We can approximate trunc by: sign(x) * floor(abs(x))
            // But Burn may not have trunc directly, so use: a - int(a/b) * b conceptually
            // For now, use the same formula but with truncation semantics:
            // fmod(a, b) = a - trunc(a/b) * b
            // trunc can be approximated as: where(x >= 0, floor(x), ceil(x))
            let positive_mask = div_result
                .clone()
                .greater_equal(Tensor::<B, 4>::zeros(div_result.dims(), device));
            let truncated = positive_mask.clone().float() * div_result.clone().floor()
                + positive_mask.bool_not().float() * div_result.ceil();
            a_tensor - truncated * b_tensor
        } else {
            // Python-style mod: a - floor(a/b) * b (sign follows divisor)
            let div_result = a_tensor.clone().div(b_tensor.clone());
            let floored = div_result.floor();
            a_tensor - floored * b_tensor
        };

        let output_rank = a.rank().max(b.rank());
        let output_dyn = rank4_to_dyn_tensor(result, output_rank)?;

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Mod node"))
    }
}

pub fn sum_variadic<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Sum(n) = node {
        let output_name = &n.outputs[0].name;

        // Sum can have multiple inputs - accumulate element-wise sum
        if n.inputs.is_empty() {
            return Err(anyhow!("Sum requires at least one input"));
        }

        let first = get_input_tensor(&n.inputs[0], values, device)?;
        let mut result = first.as_rank4();
        let mut max_rank = first.rank();

        for input in n.inputs.iter().skip(1) {
            let tensor = get_input_tensor(input, values, device)?;
            max_rank = max_rank.max(tensor.rank());
            result = result.add(tensor.as_rank4());
        }

        let output_dyn = rank4_to_dyn_tensor(result, max_rank)?;
        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Sum node"))
    }
}

pub fn mean_variadic<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Mean(n) = node {
        let output_name = &n.outputs[0].name;

        // Mean of multiple inputs: (a + b + c + ...) / n
        if n.inputs.is_empty() {
            return Err(anyhow!("Mean requires at least one input"));
        }

        let first = get_input_tensor(&n.inputs[0], values, device)?;
        let mut result = first.as_rank4();
        let mut max_rank = first.rank();

        for input in n.inputs.iter().skip(1) {
            let tensor = get_input_tensor(input, values, device)?;
            max_rank = max_rank.max(tensor.rank());
            result = result.add(tensor.as_rank4());
        }

        // Divide by number of inputs
        let count = n.inputs.len() as f32;
        result = result.div_scalar(count);

        let output_dyn = rank4_to_dyn_tensor(result, max_rank)?;
        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a Mean node"))
    }
}
