use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;

/// LayerNormalization operator
/// Normalizes the input across the last dimensions (specified by axis)
/// Formula: y = scale * (x - mean) / sqrt(variance + epsilon) + bias
pub fn layer_norm<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::LayerNormalization(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for LayerNorm", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();
        let input_rank = input_dyn.rank();
        let input_shape = input_dyn.shape().to_vec();

        // Get scale (gamma) tensor
        let scale_data = n.inputs[1]
            .value()
            .ok_or_else(|| anyhow!("LayerNorm scale tensor not found"))?;
        let scale_slice = scale_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert scale to f32: {:?}", e))?;
        let scale: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(scale_slice.to_vec(), [config.d_model]),
            device,
        );

        // Get optional bias (beta) tensor
        let bias: Option<Tensor<B, 1>> = if config.has_bias && n.inputs.len() > 2 {
            let bias_data = n.inputs[2]
                .value()
                .ok_or_else(|| anyhow!("LayerNorm bias tensor not found"))?;
            let bias_slice = bias_data
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
            Some(Tensor::from_data(
                TensorData::new(bias_slice.to_vec(), [config.d_model]),
                device,
            ))
        } else {
            None
        };

        let epsilon = config.epsilon as f32;

        // Compute mean and variance over the last dimension
        // For rank-4 tensor stored as [1, orig_d0, orig_d1, orig_d2] or similar
        // We normalize over the last axis (d_model dimension)

        // Calculate mean over last dimension
        let mean = input.clone().mean_dim(3);

        // Calculate variance: E[(x - mean)^2]
        let centered = input.clone().sub(mean.clone());
        let variance = centered.clone().powf_scalar(2.0).mean_dim(3);

        // Normalize: (x - mean) / sqrt(var + eps)
        let std = variance.add_scalar(epsilon).sqrt();
        let normalized = centered.div(std);

        // Apply scale and bias
        // Reshape scale for broadcasting: [d_model] -> [1, 1, 1, d_model]
        let scale_4d = scale.reshape([1, 1, 1, config.d_model]);
        let mut output = normalized.mul(scale_4d);

        if let Some(bias) = bias {
            let bias_4d = bias.reshape([1, 1, 1, config.d_model]);
            output = output.add(bias_4d);
        }

        // Store output with correct rank
        let output_dyn = match input_rank {
            1 => DynTensor::from_rank1(output.reshape([input_shape[0]])),
            2 => DynTensor::from_rank2(output.reshape([input_shape[0], input_shape[1]])),
            3 => DynTensor::from_rank3(output.reshape([
                input_shape[0],
                input_shape[1],
                input_shape[2],
            ])),
            4 => DynTensor::from_rank4(output),
            _ => return Err(anyhow!("Unsupported input rank for LayerNorm")),
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a LayerNormalization node"))
    }
}

/// InstanceNormalization operator
/// Normalizes each channel in each data instance independently
/// Formula: y = scale * (x - mean) / sqrt(variance + epsilon) + bias
/// where mean and variance are computed per instance per channel
pub fn instance_norm<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::InstanceNormalization(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, height, width]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for InstanceNorm", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();
        let input_shape = input_dyn.shape().to_vec();

        // Get scale tensor
        let scale_data = n.inputs[1]
            .value()
            .ok_or_else(|| anyhow!("InstanceNorm scale tensor not found"))?;
        let scale_slice = scale_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert scale to f32: {:?}", e))?;
        let scale: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(scale_slice.to_vec(), [config.num_features]),
            device,
        );

        // Get bias tensor
        let bias_data = n.inputs[2]
            .value()
            .ok_or_else(|| anyhow!("InstanceNorm bias tensor not found"))?;
        let bias_slice = bias_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
        let bias: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(bias_slice.to_vec(), [config.num_features]),
            device,
        );

        let epsilon = config.epsilon as f32;

        // Instance norm normalizes over spatial dimensions (H, W) for each channel independently
        // Mean and variance are computed over dims 2 and 3 (spatial)
        let mean = input.clone().mean_dim(2).mean_dim(3);
        let centered = input.clone().sub(mean.clone());
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2).mean_dim(3);

        // Normalize
        let std = variance.add_scalar(epsilon).sqrt();
        let normalized = centered.div(std);

        // Apply scale and bias [C] -> [1, C, 1, 1]
        let num_channels = input_shape.get(1).copied().unwrap_or(config.num_features);
        let scale_4d = scale.reshape([1, num_channels, 1, 1]);
        let bias_4d = bias.reshape([1, num_channels, 1, 1]);

        let output = normalized.mul(scale_4d).add(bias_4d);

        values.insert(output_name.clone(), DynTensor::from_rank4(output));
        Ok(())
    } else {
        Err(anyhow!("Not an InstanceNormalization node"))
    }
}

/// GroupNormalization operator
/// Normalizes input by grouping channels and computing statistics within each group
/// Formula: y = scale * (x - mean) / sqrt(variance + epsilon) + bias
pub fn group_norm<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::GroupNormalization(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, height, width]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for GroupNorm", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();
        let dims = input.dims();
        let [batch, channels, height, width] = [dims[0], dims[1], dims[2], dims[3]];

        // Get scale tensor
        let scale_data = n.inputs[1]
            .value()
            .ok_or_else(|| anyhow!("GroupNorm scale tensor not found"))?;
        let scale_slice = scale_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert scale to f32: {:?}", e))?;
        let scale: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(scale_slice.to_vec(), [config.num_features]),
            device,
        );

        // Get bias tensor
        let bias_data = n.inputs[2]
            .value()
            .ok_or_else(|| anyhow!("GroupNorm bias tensor not found"))?;
        let bias_slice = bias_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
        let bias: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(bias_slice.to_vec(), [config.num_features]),
            device,
        );

        let epsilon = config.epsilon as f32;
        let num_groups = config.num_groups;
        let channels_per_group = channels / num_groups;

        // Reshape input for group normalization:
        // [N, C, H, W] -> [N, G, C/G, H, W] (conceptually)
        // We need to normalize over (C/G, H, W) for each group

        // Reshape to [N * G, C/G * H * W] for computing stats per group
        let input_reshaped = input
            .clone()
            .reshape([batch * num_groups, channels_per_group * height * width]);

        // Compute mean and variance per group
        let mean = input_reshaped.clone().mean_dim(1); // [N * G, 1]
        let centered = input_reshaped.clone().sub(mean.clone());
        let variance = centered.clone().powf_scalar(2.0).mean_dim(1);

        // Normalize
        let std = variance.add_scalar(epsilon).sqrt();
        let normalized = centered.div(std);

        // Reshape back to [N, C, H, W]
        let normalized = normalized.reshape([batch, channels, height, width]);

        // Apply scale and bias [C] -> [1, C, 1, 1]
        let scale_4d = scale.reshape([1, channels, 1, 1]);
        let bias_4d = bias.reshape([1, channels, 1, 1]);

        let output = normalized.mul(scale_4d).add(bias_4d);

        values.insert(output_name.clone(), DynTensor::from_rank4(output));
        Ok(())
    } else {
        Err(anyhow!("Not a GroupNormalization node"))
    }
}
