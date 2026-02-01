use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::activation;
use burn::tensor::Tensor;
use onnx_ir::ir::Node;

pub fn relu<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Relu(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Relu", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        let output_tensor = activation::relu(input_tensor);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Relu node"))
    }
}

pub fn sigmoid<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Sigmoid(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Sigmoid", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        let output_tensor = activation::sigmoid(input_tensor);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Sigmoid node"))
    }
}

pub fn softmax<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Softmax(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Softmax", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        // Map axis to rank-4 tensor dimension
        let rank4_axis = axis + (4 - input_dyn.rank());
        let output_tensor = activation::softmax(input_tensor, rank4_axis);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Softmax node"))
    }
}

pub fn tanh<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Tanh(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Tanh", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        let output_tensor = activation::tanh(input_tensor);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Tanh node"))
    }
}

pub fn gelu<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Gelu(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Gelu", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        let output_tensor = activation::gelu(input_tensor);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Gelu node"))
    }
}

pub fn leaky_relu<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::LeakyRelu(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let alpha = n.config.alpha;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for LeakyRelu", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        let output_tensor = activation::leaky_relu(input_tensor, alpha);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a LeakyRelu node"))
    }
}

pub fn hard_sigmoid<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::HardSigmoid(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let alpha = n.config.alpha as f32;
        let beta = n.config.beta as f32;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for HardSigmoid", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        // HardSigmoid: y = max(0, min(1, alpha * x + beta))
        let output_tensor = (input_tensor.mul_scalar(alpha).add_scalar(beta)).clamp(0.0, 1.0);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a HardSigmoid node"))
    }
}

pub fn hard_swish<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::HardSwish(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for HardSwish", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        // HardSwish: y = x * max(0, min(1, (x + 3) / 6))
        let hard_sigmoid_part =
            (input_tensor.clone().add_scalar(3.0).div_scalar(6.0)).clamp(0.0, 1.0);
        let output_tensor = input_tensor.mul(hard_sigmoid_part);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a HardSwish node"))
    }
}

pub fn prelu<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::PRelu(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for PRelu", input_name))?;

        // Get slope from input[1] (static value)
        let slope_arg = &n.inputs[1];
        let slope_data = slope_arg
            .value()
            .ok_or_else(|| anyhow!("PRelu slope must be a static value"))?;
        let slope_slice = slope_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not read slope tensor: {:?}", e))?;

        let input_tensor = input_dyn.as_rank4();

        // PRelu: y = max(0, x) + slope * min(0, x)
        // If slope is scalar, use it directly
        if slope_slice.len() == 1 {
            let alpha = slope_slice[0] as f64;
            let output_tensor = activation::leaky_relu(input_tensor, alpha);
            values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        } else {
            // For per-channel slope, we need to broadcast
            // Create slope tensor with proper shape for broadcasting
            use burn::tensor::TensorData;
            let slope_shape = slope_arg
                .ty
                .static_shape()
                .map(|s| s.clone())
                .unwrap_or_else(|| vec![slope_slice.len()]);

            // Pad slope shape to rank 4
            let mut padded_shape = vec![1usize; 4];
            let offset = 4usize.saturating_sub(slope_shape.len());
            for (i, &dim) in slope_shape.iter().enumerate() {
                if offset + i < 4 {
                    padded_shape[offset + i] = dim;
                }
            }

            let slope_tensor_data = TensorData::new(slope_slice.to_vec(), padded_shape);
            let slope_tensor: Tensor<B, 4> = Tensor::from_data(slope_tensor_data, device);

            // PRelu formula: y = max(0, x) + slope * min(0, x)
            let pos_part = input_tensor.clone().clamp_min(0.0);
            let neg_part = input_tensor.clamp_max(0.0).mul(slope_tensor);
            let output_tensor = pos_part.add(neg_part);

            values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        }
        Ok(())
    } else {
        Err(anyhow!("Not a PRelu node"))
    }
}

pub fn log_softmax<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::LogSoftmax(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let axis = n.config.axis;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for LogSoftmax", input_name))?;

        let input_tensor = input_dyn.as_rank4();
        // Map axis to rank-4 tensor dimension
        let rank4_axis = axis + (4 - input_dyn.rank());
        let output_tensor = activation::log_softmax(input_tensor, rank4_axis);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a LogSoftmax node"))
    }
}

pub fn identity_<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Identity(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Identity", input_name))?;

        // Identity just passes through the tensor unchanged
        values.insert(output_name.clone(), input_dyn.clone());
        Ok(())
    } else {
        Err(anyhow!("Not an Identity node"))
    }
}

pub fn dropout<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Dropout(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Dropout", input_name))?;

        // Dropout is a no-op in inference - pass through unchanged
        values.insert(output_name.clone(), input_dyn.clone());
        Ok(())
    } else {
        Err(anyhow!("Not a Dropout node"))
    }
}

pub fn clip<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Clip(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Clip", input_name))?;

        // Get min/max from node inputs (may be static or in value store)
        let min_val = if n.inputs.len() > 1 {
            if let Some(v) = n.inputs[1].value() {
                if let Ok(s) = v.as_slice::<f32>() {
                    s[0]
                } else {
                    f32::NEG_INFINITY
                }
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NEG_INFINITY
        };

        let max_val = if n.inputs.len() > 2 {
            if let Some(v) = n.inputs[2].value() {
                if let Ok(s) = v.as_slice::<f32>() {
                    s[0]
                } else {
                    f32::INFINITY
                }
            } else {
                f32::INFINITY
            }
        } else {
            f32::INFINITY
        };

        let input_tensor = input_dyn.as_rank4();
        let output_tensor = input_tensor.clamp(min_val, max_val);

        values.insert(output_name.clone(), DynTensor::from_rank4(output_tensor));
        Ok(())
    } else {
        Err(anyhow!("Not a Clip node"))
    }
}

/// Cast operation - converts tensor between dtypes
/// Note: Currently nnx only supports float tensors internally, so Cast operations
/// between float types are treated as no-ops. Cast to/from int/bool types
/// will require enhanced DynTensor support.
pub fn cast<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::Cast(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let target_dtype = &n.config.to;

        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Cast", input_name))?;

        // For now, DynTensor only supports float tensors internally
        // Cast between float types is a no-op (F16, F32, F64 all stored as F32)
        // For int/bool types, we pass through but note the limitation
        use burn::tensor::DType;
        match target_dtype {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                // Float-to-float cast: pass through (no-op in our current representation)
                values.insert(output_name.clone(), input_dyn.clone());
            }
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                // Float-to-int cast: for now, pass through with a warning
                // This works if the tensor is used in contexts expecting floats
                // but may cause issues if integer operations are expected
                values.insert(output_name.clone(), input_dyn.clone());
            }
            DType::Bool => {
                // Float-to-bool cast: pass through
                // Non-zero values are true, zero is false
                values.insert(output_name.clone(), input_dyn.clone());
            }
            _ => {
                return Err(anyhow!("Cast: unsupported target dtype {:?}", target_dtype));
            }
        }
        Ok(())
    } else {
        Err(anyhow!("Not a Cast node"))
    }
}

/// ConstantOfShape - creates a tensor filled with a constant value
pub fn constant_of_shape<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    use burn::tensor::TensorData;

    if let Node::ConstantOfShape(n) = node {
        let output_name = &n.outputs[0].name;

        // Get the shape from config
        use onnx_ir::node::constant_of_shape::ConstantOfShapeShape;

        let shape: Vec<usize> = match &n.config.shape {
            ConstantOfShapeShape::Static(s) => s.iter().map(|&d| d as usize).collect(),
            ConstantOfShapeShape::Runtime(_) => {
                return Err(anyhow!("ConstantOfShape: runtime shape not yet supported"));
            }
        };

        // Get the fill value (default is 0.0f32)
        let fill_value: f32 = if let Some(ref value_data) = n.config.value {
            // Try to extract value as f32
            if let Ok(s) = value_data.as_slice::<f32>() {
                if !s.is_empty() {
                    s[0]
                } else {
                    0.0
                }
            } else if let Ok(s) = value_data.as_slice::<i64>() {
                if !s.is_empty() {
                    s[0] as f32
                } else {
                    0.0
                }
            } else if let Ok(s) = value_data.as_slice::<i32>() {
                if !s.is_empty() {
                    s[0] as f32
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let rank = shape.len();

        // Pad shape to rank 4
        let mut padded_shape = [1usize; 4];
        for (i, &dim) in shape.iter().enumerate() {
            padded_shape[4 - rank + i] = dim;
        }

        // Create the tensor filled with constant value
        let output_tensor: Tensor<B, 4> = Tensor::full(padded_shape, fill_value, device);

        let output_dyn = match rank {
            1 => {
                let t: Tensor<B, 1> = output_tensor.reshape([shape[0]]);
                DynTensor::from_rank1(t)
            }
            2 => {
                let t: Tensor<B, 2> = output_tensor.reshape([shape[0], shape[1]]);
                DynTensor::from_rank2(t)
            }
            3 => {
                let t: Tensor<B, 3> = output_tensor.reshape([shape[0], shape[1], shape[2]]);
                DynTensor::from_rank3(t)
            }
            4 => DynTensor::from_rank4(output_tensor),
            0 => {
                // Scalar (rank 0) - create as rank 1 with single element
                let t: Tensor<B, 1> = Tensor::full([1], fill_value, device);
                DynTensor::from_rank1(t)
            }
            _ => {
                return Err(anyhow!("ConstantOfShape: unsupported output rank {}", rank));
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a ConstantOfShape node"))
    }
}
