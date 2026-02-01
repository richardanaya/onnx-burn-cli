use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use onnx_ir::ir::Node;

pub fn abs<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Abs(_n) = node {
        unary_op(node, values, |t| t.abs())
    } else {
        Err(anyhow!("Not an Abs node"))
    }
}

pub fn neg<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Neg(_n) = node {
        unary_op(node, values, |t| t.neg())
    } else {
        Err(anyhow!("Not a Neg node"))
    }
}

pub fn sqrt<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Sqrt(_n) = node {
        unary_op(node, values, |t| t.sqrt())
    } else {
        Err(anyhow!("Not a Sqrt node"))
    }
}

pub fn exp<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Exp(_n) = node {
        unary_op(node, values, |t| t.exp())
    } else {
        Err(anyhow!("Not an Exp node"))
    }
}

pub fn log<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Log(_n) = node {
        unary_op(node, values, |t| t.log())
    } else {
        Err(anyhow!("Not a Log node"))
    }
}

pub fn ceil<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Ceil(_n) = node {
        unary_op(node, values, |t| t.ceil())
    } else {
        Err(anyhow!("Not a Ceil node"))
    }
}

pub fn floor<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Floor(_n) = node {
        unary_op(node, values, |t| t.floor())
    } else {
        Err(anyhow!("Not a Floor node"))
    }
}

pub fn round<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Round(_n) = node {
        unary_op(node, values, |t| t.round())
    } else {
        Err(anyhow!("Not a Round node"))
    }
}

pub fn sign<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Sign(_n) = node {
        unary_op(node, values, |t| t.sign())
    } else {
        Err(anyhow!("Not a Sign node"))
    }
}

pub fn reciprocal<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Reciprocal(_n) = node {
        unary_op(node, values, |t| t.recip())
    } else {
        Err(anyhow!("Not a Reciprocal node"))
    }
}

pub fn sin<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Sin(_n) = node {
        unary_op(node, values, |t| t.sin())
    } else {
        Err(anyhow!("Not a Sin node"))
    }
}

pub fn cos<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Cos(_n) = node {
        unary_op(node, values, |t| t.cos())
    } else {
        Err(anyhow!("Not a Cos node"))
    }
}

pub fn tan<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Tan(_n) = node {
        unary_op(node, values, |t| t.tan())
    } else {
        Err(anyhow!("Not a Tan node"))
    }
}

pub fn sinh<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Sinh(_n) = node {
        unary_op(node, values, |t| t.sinh())
    } else {
        Err(anyhow!("Not a Sinh node"))
    }
}

pub fn cosh<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Cosh(_n) = node {
        unary_op(node, values, |t| t.cosh())
    } else {
        Err(anyhow!("Not a Cosh node"))
    }
}

pub fn erf<B: Backend>(node: &Node, values: &mut ValueStore<B>, _device: &B::Device) -> Result<()> {
    if let Node::Erf(_n) = node {
        unary_op(node, values, |t| t.erf())
    } else {
        Err(anyhow!("Not an Erf node"))
    }
}

pub fn atan<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Atan(_n) = node {
        unary_op(node, values, |t| t.atan())
    } else {
        Err(anyhow!("Not an Atan node"))
    }
}

/// Generic unary operation handler
fn unary_op<F, B: Backend>(node: &Node, values: &mut ValueStore<B>, op: F) -> Result<()>
where
    F: FnOnce(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let input_name = &node.inputs()[0].name;
    let output_name = &node.outputs()[0].name;

    let input_dyn = values
        .get(input_name)
        .ok_or_else(|| anyhow!("Input tensor '{}' not found", input_name))?;

    let input_tensor = input_dyn.as_rank4();
    let output_tensor = op(input_tensor);

    values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));

    Ok(())
}
