use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use onnx_ir::ir::Node;

/// Helper to perform binary comparison and store result
fn comparison_op<B: Backend, F>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
    op_name: &str,
    compare_fn: F,
) -> Result<()>
where
    F: FnOnce(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
{
    let inputs = node.inputs();
    let outputs = node.outputs();

    let input1_name = &inputs[0].name;
    let input2_name = &inputs[1].name;
    let output_name = &outputs[0].name;

    let input1_dyn = values
        .get(input1_name)
        .ok_or_else(|| anyhow!("Input tensor '{}' not found for {}", input1_name, op_name))?;
    let input2_dyn = values
        .get(input2_name)
        .ok_or_else(|| anyhow!("Input tensor '{}' not found for {}", input2_name, op_name))?;

    let input1_4d = input1_dyn.as_rank4();
    let input2_4d = input2_dyn.as_rank4();

    // Comparison operations return bool tensors, but Burn returns int tensors (0 or 1)
    // We'll keep them as float for now since DynTensor uses float tensors
    let result = compare_fn(input1_4d, input2_4d);

    // Output rank is max of input ranks
    let output_rank = input1_dyn.rank().max(input2_dyn.rank());
    let output_shape: Vec<usize> = if output_rank == input1_dyn.rank() {
        input1_dyn.shape().to_vec()
    } else {
        input2_dyn.shape().to_vec()
    };

    let output_dyn = match output_rank {
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
        _ => {
            return Err(anyhow!(
                "{}: unsupported output rank {}",
                op_name,
                output_rank
            ))
        }
    };

    values.insert(output_name.clone(), output_dyn);
    Ok(())
}

pub fn equal<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Equal(_) = node {
        comparison_op(node, values, device, "Equal", |a, b| a.equal(b).float())
    } else {
        Err(anyhow!("Not an Equal node"))
    }
}

pub fn greater<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Greater(_) = node {
        comparison_op(node, values, device, "Greater", |a, b| a.greater(b).float())
    } else {
        Err(anyhow!("Not a Greater node"))
    }
}

pub fn less<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Less(_) = node {
        comparison_op(node, values, device, "Less", |a, b| a.lower(b).float())
    } else {
        Err(anyhow!("Not a Less node"))
    }
}

pub fn greater_or_equal<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::GreaterOrEqual(_) = node {
        comparison_op(node, values, device, "GreaterOrEqual", |a, b| {
            a.greater_equal(b).float()
        })
    } else {
        Err(anyhow!("Not a GreaterOrEqual node"))
    }
}

pub fn less_or_equal<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::LessOrEqual(_) = node {
        comparison_op(node, values, device, "LessOrEqual", |a, b| {
            a.lower_equal(b).float()
        })
    } else {
        Err(anyhow!("Not a LessOrEqual node"))
    }
}

pub fn is_inf<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::IsInf(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for IsInf", input_name))?;

        let input_4d = input_dyn.as_rank4();

        // Check for infinity - burn doesn't have is_inf, so we use comparison with infinity
        // is_inf(x) = (x == inf) or (x == -inf)
        let pos_inf = input_4d.clone().equal(
            input_4d
                .clone()
                .mul_scalar(f32::INFINITY / f32::INFINITY * f32::INFINITY),
        );
        let neg_inf = input_4d
            .clone()
            .equal(input_4d.mul_scalar(f32::NEG_INFINITY / f32::INFINITY * f32::INFINITY));
        let result = pos_inf.float().add(neg_inf.float()).clamp(0.0, 1.0);

        let rank = input_dyn.rank();
        let shape = input_dyn.shape().to_vec();

        let output_dyn = match rank {
            1 => {
                let output: Tensor<B, 1> = result.reshape([shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = result.reshape([shape[0], shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> = result.reshape([shape[0], shape[1], shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(result),
            _ => return Err(anyhow!("IsInf: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an IsInf node"))
    }
}

pub fn is_nan<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::IsNaN(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for IsNaN", input_name))?;

        let input_4d = input_dyn.as_rank4();

        // Check for NaN - NaN is the only value that is not equal to itself
        let result = input_4d.clone().equal(input_4d).bool_not().float();

        let rank = input_dyn.rank();
        let shape = input_dyn.shape().to_vec();

        let output_dyn = match rank {
            1 => {
                let output: Tensor<B, 1> = result.reshape([shape[0]]);
                DynTensor::from_rank1(output)
            }
            2 => {
                let output: Tensor<B, 2> = result.reshape([shape[0], shape[1]]);
                DynTensor::from_rank2(output)
            }
            3 => {
                let output: Tensor<B, 3> = result.reshape([shape[0], shape[1], shape[2]]);
                DynTensor::from_rank3(output)
            }
            4 => DynTensor::from_rank4(result),
            _ => return Err(anyhow!("IsNaN: unsupported rank {}", rank)),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not an IsNaN node"))
    }
}
