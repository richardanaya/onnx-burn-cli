use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{module, ops::ConvOptions, ops::ConvTransposeOptions, Tensor, TensorData};
use onnx_ir::ir::Node;
use onnx_ir::padding::{PaddingConfig1d, PaddingConfig2d};

/// Convert ONNX PaddingConfig2d to burn's padding array
fn padding_to_array(padding: &PaddingConfig2d) -> [usize; 2] {
    match padding {
        PaddingConfig2d::Explicit(h, w) => [*h, *w],
        PaddingConfig2d::Valid => [0, 0],
    }
}

/// Convert ONNX PaddingConfig1d to burn's padding value
fn padding_1d_to_value(padding: &PaddingConfig1d) -> usize {
    match padding {
        PaddingConfig1d::Explicit(p) => *p,
        PaddingConfig1d::Valid => 0,
    }
}

pub fn conv2d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Conv2d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels_in, height, width]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Conv2d", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();

        // Get weight tensor [channels_out, channels_in/groups, kernel_h, kernel_w]
        let weight_arg = &n.inputs[1];
        let weight_data = weight_arg
            .value()
            .ok_or_else(|| anyhow!("Conv2d weight tensor not found"))?;
        let weight_slice = weight_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert weight to f32: {:?}", e))?;

        let [channels_in, channels_out] = config.channels;
        let [kernel_h, kernel_w] = config.kernel_size;
        let channels_in_per_group = channels_in / config.groups;

        let weight_tensor_data = TensorData::new(
            weight_slice.to_vec(),
            [channels_out, channels_in_per_group, kernel_h, kernel_w],
        );
        let weight: Tensor<B, 4> = Tensor::from_data(weight_tensor_data, device);

        // Get optional bias [channels_out]
        let bias: Option<Tensor<B, 1>> = if config.bias && n.inputs.len() > 2 {
            let bias_arg = &n.inputs[2];
            if let Some(bias_data) = bias_arg.value() {
                let bias_slice = bias_data
                    .as_slice::<f32>()
                    .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
                let bias_tensor_data = TensorData::new(bias_slice.to_vec(), [channels_out]);
                Some(Tensor::from_data(bias_tensor_data, device))
            } else {
                None
            }
        } else {
            None
        };

        // Build ConvOptions
        let padding = padding_to_array(&config.padding);
        let conv_options = ConvOptions::new(config.stride, padding, config.dilation, config.groups);

        // Execute convolution
        let output = module::conv2d(input, weight, bias, conv_options);

        // Store output
        values.insert(output_name.clone(), DynTensor::from_rank4(output));

        Ok(())
    } else {
        Err(anyhow!("Not a Conv2d node"))
    }
}

pub fn max_pool_2d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::MaxPool2d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, height, width]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for MaxPool2d", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();

        // Build pool options
        let padding = padding_to_array(&config.padding);

        // Execute max pooling using burn's module function
        // The last parameter is ceil_mode (whether to use ceiling when computing output size)
        let output = module::max_pool2d(
            input,
            config.kernel_size,
            config.strides,
            padding,
            config.dilation,
            config.ceil_mode,
        );

        // Store output
        values.insert(output_name.clone(), DynTensor::from_rank4(output));

        Ok(())
    } else {
        Err(anyhow!("Not a MaxPool2d node"))
    }
}

pub fn batch_normalization<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::BatchNormalization(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, height, width]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for BatchNorm", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();

        // BatchNorm inputs:
        // 0: input
        // 1: scale (gamma)
        // 2: bias (beta)
        // 3: running_mean
        // 4: running_var

        let num_features = config.num_features;
        let epsilon = config.epsilon as f32;

        // Get scale (gamma)
        let scale_data = n.inputs[1]
            .value()
            .ok_or_else(|| anyhow!("BatchNorm scale not found"))?;
        let scale_slice = scale_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert scale to f32: {:?}", e))?;
        let scale: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(scale_slice.to_vec(), [num_features]),
            device,
        );

        // Get bias (beta)
        let bias_data = n.inputs[2]
            .value()
            .ok_or_else(|| anyhow!("BatchNorm bias not found"))?;
        let bias_slice = bias_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
        let bias: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(bias_slice.to_vec(), [num_features]), device);

        // Get running mean
        let mean_data = n.inputs[3]
            .value()
            .ok_or_else(|| anyhow!("BatchNorm running_mean not found"))?;
        let mean_slice = mean_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert mean to f32: {:?}", e))?;
        let running_mean: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(mean_slice.to_vec(), [num_features]), device);

        // Get running variance
        let var_data = n.inputs[4]
            .value()
            .ok_or_else(|| anyhow!("BatchNorm running_var not found"))?;
        let var_slice = var_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert var to f32: {:?}", e))?;
        let running_var: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(var_slice.to_vec(), [num_features]), device);

        // Apply batch normalization formula for inference:
        // output = (input - mean) / sqrt(var + epsilon) * scale + bias
        //
        // Reshape for broadcasting: [1, C, 1, 1]
        let mean_4d = running_mean.reshape([1, num_features, 1, 1]);
        let var_4d = running_var.reshape([1, num_features, 1, 1]);
        let scale_4d = scale.reshape([1, num_features, 1, 1]);
        let bias_4d = bias.reshape([1, num_features, 1, 1]);

        // Compute: (input - mean) / sqrt(var + eps) * scale + bias
        let normalized = input.sub(mean_4d);
        let std = var_4d.add_scalar(epsilon).sqrt();
        let output = normalized.div(std).mul(scale_4d).add(bias_4d);

        // Store output
        values.insert(output_name.clone(), DynTensor::from_rank4(output));

        Ok(())
    } else {
        Err(anyhow!("Not a BatchNormalization node"))
    }
}

pub fn global_average_pool<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::GlobalAveragePool(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        // Get input tensor [batch, channels, height, width]
        let input_dyn = values.get(input_name).ok_or_else(|| {
            anyhow!(
                "Input tensor '{}' not found for GlobalAveragePool",
                input_name
            )
        })?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();

        // Global average pooling: reduce spatial dimensions to 1x1
        // by averaging over height and width
        // Use adaptive_avg_pool2d with output size [1, 1]
        let output = module::adaptive_avg_pool2d(input, [1, 1]);

        // Output shape should be [batch, channels, 1, 1]
        // Store as rank-4 tensor
        values.insert(output_name.clone(), DynTensor::from_rank4(output));

        Ok(())
    } else {
        Err(anyhow!("Not a GlobalAveragePool node"))
    }
}

pub fn avg_pool_2d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::AveragePool2d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, height, width]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for AvgPool2d", input_name))?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();

        // Build pool options
        let padding = padding_to_array(&config.padding);

        // Execute average pooling using burn's module function
        // Note: count_include_pad controls whether padding elements are included in average
        let output = module::avg_pool2d(
            input,
            config.kernel_size,
            config.strides,
            padding,
            config.count_include_pad,
            config.ceil_mode,
        );

        // Store output
        values.insert(output_name.clone(), DynTensor::from_rank4(output));

        Ok(())
    } else {
        Err(anyhow!("Not an AvgPool2d node"))
    }
}

pub fn avg_pool_1d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::AveragePool1d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, length]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for AvgPool1d", input_name))?;
        let input: Tensor<B, 3> = input_dyn.as_rank3();

        // Build pool options
        let padding = padding_1d_to_value(&config.padding);

        // Execute average pooling using burn's module function
        let output = module::avg_pool1d(
            input,
            config.kernel_size,
            config.stride,
            padding,
            config.count_include_pad,
            config.ceil_mode,
        );

        // Store output as rank-3 tensor
        values.insert(output_name.clone(), DynTensor::from_rank3(output));

        Ok(())
    } else {
        Err(anyhow!("Not an AvgPool1d node"))
    }
}

pub fn max_pool_1d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    _device: &B::Device,
) -> Result<()> {
    if let Node::MaxPool1d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels, length]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for MaxPool1d", input_name))?;
        let input: Tensor<B, 3> = input_dyn.as_rank3();

        // Build pool options
        let padding = padding_1d_to_value(&config.padding);

        // Execute max pooling using burn's module function
        let output = module::max_pool1d(
            input,
            config.kernel_size,
            config.stride,
            padding,
            config.dilation,
            config.ceil_mode,
        );

        // Store output as rank-3 tensor
        values.insert(output_name.clone(), DynTensor::from_rank3(output));

        Ok(())
    } else {
        Err(anyhow!("Not a MaxPool1d node"))
    }
}

pub fn conv1d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Conv1d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels_in, length]
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for Conv1d", input_name))?;
        let input: Tensor<B, 3> = input_dyn.as_rank3();

        // Get weight tensor [channels_out, channels_in/groups, kernel_size]
        let weight_arg = &n.inputs[1];
        let weight_data = weight_arg
            .value()
            .ok_or_else(|| anyhow!("Conv1d weight tensor not found"))?;
        let weight_slice = weight_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert weight to f32: {:?}", e))?;

        let channels_in_per_group = config.channels_in / config.groups;

        let weight_tensor_data = TensorData::new(
            weight_slice.to_vec(),
            [
                config.channels_out,
                channels_in_per_group,
                config.kernel_size,
            ],
        );
        let weight: Tensor<B, 3> = Tensor::from_data(weight_tensor_data, device);

        // Get optional bias [channels_out]
        let bias: Option<Tensor<B, 1>> = if config.bias && n.inputs.len() > 2 {
            let bias_arg = &n.inputs[2];
            if let Some(bias_data) = bias_arg.value() {
                let bias_slice = bias_data
                    .as_slice::<f32>()
                    .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
                let bias_tensor_data = TensorData::new(bias_slice.to_vec(), [config.channels_out]);
                Some(Tensor::from_data(bias_tensor_data, device))
            } else {
                None
            }
        } else {
            None
        };

        // Build ConvOptions
        let padding = padding_1d_to_value(&config.padding);
        let conv_options =
            ConvOptions::new([config.stride], [padding], [config.dilation], config.groups);

        // Execute convolution
        let output = module::conv1d(input, weight, bias, conv_options);

        // Store output as rank-3 tensor
        values.insert(output_name.clone(), DynTensor::from_rank3(output));

        Ok(())
    } else {
        Err(anyhow!("Not a Conv1d node"))
    }
}

pub fn conv_transpose2d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::ConvTranspose2d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels_in, height, width]
        let input_dyn = values.get(input_name).ok_or_else(|| {
            anyhow!(
                "Input tensor '{}' not found for ConvTranspose2d",
                input_name
            )
        })?;
        let input: Tensor<B, 4> = input_dyn.as_rank4();

        // Get weight tensor [channels_in, channels_out/groups, kernel_h, kernel_w]
        // Note: ConvTranspose weight layout is different from Conv
        let weight_arg = &n.inputs[1];
        let weight_data = weight_arg
            .value()
            .ok_or_else(|| anyhow!("ConvTranspose2d weight tensor not found"))?;
        let weight_slice = weight_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert weight to f32: {:?}", e))?;

        let [channels_in, channels_out] = config.channels;
        let channels_out_per_group = channels_out / config.groups;
        let [kernel_h, kernel_w] = config.kernel_size;

        let weight_tensor_data = TensorData::new(
            weight_slice.to_vec(),
            [channels_in, channels_out_per_group, kernel_h, kernel_w],
        );
        let weight: Tensor<B, 4> = Tensor::from_data(weight_tensor_data, device);

        // Get optional bias [channels_out]
        let bias: Option<Tensor<B, 1>> = if config.bias && n.inputs.len() > 2 {
            let bias_arg = &n.inputs[2];
            if let Some(bias_data) = bias_arg.value() {
                let bias_slice = bias_data
                    .as_slice::<f32>()
                    .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
                let bias_tensor_data = TensorData::new(bias_slice.to_vec(), [channels_out]);
                Some(Tensor::from_data(bias_tensor_data, device))
            } else {
                None
            }
        } else {
            None
        };

        // Build ConvTransposeOptions
        let conv_options = ConvTransposeOptions::new(
            config.stride,
            config.padding,
            config.padding_out,
            config.dilation,
            config.groups,
        );

        // Execute transposed convolution
        let output = module::conv_transpose2d(input, weight, bias, conv_options);

        // Store output
        values.insert(output_name.clone(), DynTensor::from_rank4(output));

        Ok(())
    } else {
        Err(anyhow!("Not a ConvTranspose2d node"))
    }
}

pub fn conv_transpose1d<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::ConvTranspose1d(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        // Get input tensor [batch, channels_in, length]
        let input_dyn = values.get(input_name).ok_or_else(|| {
            anyhow!(
                "Input tensor '{}' not found for ConvTranspose1d",
                input_name
            )
        })?;
        let input: Tensor<B, 3> = input_dyn.as_rank3();

        // Get weight tensor [channels_in, channels_out/groups, kernel_size]
        let weight_arg = &n.inputs[1];
        let weight_data = weight_arg
            .value()
            .ok_or_else(|| anyhow!("ConvTranspose1d weight tensor not found"))?;
        let weight_slice = weight_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert weight to f32: {:?}", e))?;

        let channels_out_per_group = config.channels_out / config.groups;

        let weight_tensor_data = TensorData::new(
            weight_slice.to_vec(),
            [
                config.channels_in,
                channels_out_per_group,
                config.kernel_size,
            ],
        );
        let weight: Tensor<B, 3> = Tensor::from_data(weight_tensor_data, device);

        // Get optional bias [channels_out]
        let bias: Option<Tensor<B, 1>> = if config.bias && n.inputs.len() > 2 {
            let bias_arg = &n.inputs[2];
            if let Some(bias_data) = bias_arg.value() {
                let bias_slice = bias_data
                    .as_slice::<f32>()
                    .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
                let bias_tensor_data = TensorData::new(bias_slice.to_vec(), [config.channels_out]);
                Some(Tensor::from_data(bias_tensor_data, device))
            } else {
                None
            }
        } else {
            None
        };

        // Build ConvTransposeOptions
        let conv_options = ConvTransposeOptions::new(
            [config.stride],
            [config.padding],
            [config.padding_out],
            [config.dilation],
            config.groups,
        );

        // Execute transposed convolution
        let output = module::conv_transpose1d(input, weight, bias, conv_options);

        // Store output as rank-3 tensor
        values.insert(output_name.clone(), DynTensor::from_rank3(output));

        Ok(())
    } else {
        Err(anyhow!("Not a ConvTranspose1d node"))
    }
}

/// Generic ConvTranspose handler for unsupported ConvTranspose nodes
/// Examines input rank to dispatch to ConvTranspose1d or ConvTranspose2d
pub fn conv_transpose<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::ConvTranspose(n) = node {
        let input_name = &n.inputs[0].name;
        let output_name = &n.outputs[0].name;

        // Get input tensor to determine rank
        let input_dyn = values
            .get(input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found for ConvTranspose", input_name))?;

        let input_rank = input_dyn.rank();

        // Get weight tensor to determine kernel dimensions
        let weight_arg = &n.inputs[1];
        let weight_data = weight_arg
            .value()
            .ok_or_else(|| anyhow!("ConvTranspose weight tensor not found"))?;
        let weight_slice = weight_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert weight to f32: {:?}", e))?;
        let weight_shape: Vec<usize> = weight_data.shape.iter().map(|&d| d as usize).collect();

        match input_rank {
            3 => {
                // ConvTranspose1d: input [batch, channels_in, length]
                // Weight: [channels_in, channels_out/groups, kernel_size]
                let input: Tensor<B, 3> = input_dyn.as_rank3();

                let channels_in = weight_shape[0];
                let channels_out_per_group = weight_shape[1];
                let kernel_size = weight_shape[2];

                // Infer groups and channels_out from weight shape
                // Typically groups=1 for most models
                let groups = 1usize;
                let channels_out = channels_out_per_group * groups;

                let weight_tensor_data = TensorData::new(
                    weight_slice.to_vec(),
                    [channels_in, channels_out_per_group, kernel_size],
                );
                let weight: Tensor<B, 3> = Tensor::from_data(weight_tensor_data, device);

                // Get optional bias
                let bias: Option<Tensor<B, 1>> = if n.inputs.len() > 2 {
                    if let Some(bias_data) = n.inputs[2].value() {
                        let bias_slice = bias_data
                            .as_slice::<f32>()
                            .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
                        let bias_tensor_data = TensorData::new(bias_slice.to_vec(), [channels_out]);
                        Some(Tensor::from_data(bias_tensor_data, device))
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Default options - stride=1, padding=0, dilation=1
                // TODO: Extract from node attributes if available
                let conv_options = ConvTransposeOptions::new([1], [0], [0], [1], groups);

                let output = module::conv_transpose1d(input, weight, bias, conv_options);
                values.insert(output_name.clone(), DynTensor::from_rank3(output));
            }
            4 => {
                // ConvTranspose2d: input [batch, channels_in, height, width]
                // Weight: [channels_in, channels_out/groups, kernel_h, kernel_w]
                let input: Tensor<B, 4> = input_dyn.as_rank4();

                let channels_in = weight_shape[0];
                let channels_out_per_group = weight_shape[1];
                let kernel_h = weight_shape[2];
                let kernel_w = weight_shape[3];

                let groups = 1usize;
                let channels_out = channels_out_per_group * groups;

                let weight_tensor_data = TensorData::new(
                    weight_slice.to_vec(),
                    [channels_in, channels_out_per_group, kernel_h, kernel_w],
                );
                let weight: Tensor<B, 4> = Tensor::from_data(weight_tensor_data, device);

                // Get optional bias
                let bias: Option<Tensor<B, 1>> = if n.inputs.len() > 2 {
                    if let Some(bias_data) = n.inputs[2].value() {
                        let bias_slice = bias_data
                            .as_slice::<f32>()
                            .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
                        let bias_tensor_data = TensorData::new(bias_slice.to_vec(), [channels_out]);
                        Some(Tensor::from_data(bias_tensor_data, device))
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Default options
                let conv_options =
                    ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], groups);

                let output = module::conv_transpose2d(input, weight, bias, conv_options);
                values.insert(output_name.clone(), DynTensor::from_rank4(output));
            }
            _ => {
                return Err(anyhow!(
                    "ConvTranspose: unsupported input rank {}, expected 3 (1D) or 4 (2D)",
                    input_rank
                ));
            }
        }

        Ok(())
    } else {
        Err(anyhow!("Not a ConvTranspose node"))
    }
}
