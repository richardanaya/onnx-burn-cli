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

/// MatMul operator - matrix multiplication with broadcasting support
/// Supports: 2D x 2D, batched matmul, and vector-matrix multiplications
pub fn matmul<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::MatMul(n) = node {
        let output_name = &n.outputs[0].name;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        let a_rank = a.rank();
        let b_rank = b.rank();

        // Handle different rank combinations
        let output_dyn = match (a_rank, b_rank) {
            // 2D x 2D -> 2D
            (2, 2) => {
                let a_2d = a.as_rank2();
                let b_2d = b.as_rank2();
                let result = a_2d.matmul(b_2d);
                DynTensor::from_rank2(result)
            }
            // 1D x 2D -> 1D (vector-matrix)
            (1, 2) => {
                // Treat 1D as row vector: [K] -> [1, K]
                let a_shape = a.shape();
                let a_2d: Tensor<B, 2> = a.as_rank4().reshape([1, a_shape[0]]);
                let b_2d = b.as_rank2();
                let result = a_2d.matmul(b_2d);
                // Result is [1, N], squeeze to [N]
                let result_shape = result.dims();
                DynTensor::from_rank1(result.reshape([result_shape[1]]))
            }
            // 2D x 1D -> 1D (matrix-vector)
            (2, 1) => {
                // Treat 1D as column vector: [K] -> [K, 1]
                let b_shape = b.shape();
                let a_2d = a.as_rank2();
                let b_2d: Tensor<B, 2> = b.as_rank4().reshape([b_shape[0], 1]);
                let result = a_2d.matmul(b_2d);
                // Result is [M, 1], squeeze to [M]
                let result_shape = result.dims();
                DynTensor::from_rank1(result.reshape([result_shape[0]]))
            }
            // 3D x 2D -> 3D (batched, B broadcasts)
            (3, 2) => {
                let a_4d = a.as_rank4();
                let a_shape = a.shape();
                let b_2d = b.as_rank2();
                let b_shape = b.shape();

                // Reshape A from [1, batch, M, K] to [batch, M, K] and B to [batch, K, N] by broadcasting
                let batch = a_shape[0];
                let m = a_shape[1];
                let k = a_shape[2];
                let _n = b_shape[1];

                // We need to expand B to have a batch dimension
                // [K, N] -> [batch, K, N]
                let b_expanded = b_2d.unsqueeze_dim::<3>(0).repeat_dim(0, batch);
                let a_3d: Tensor<B, 3> = a_4d.reshape([batch, m, k]);

                let result = a_3d.matmul(b_expanded);
                DynTensor::from_rank3(result)
            }
            // 2D x 3D -> 3D (batched, A broadcasts)
            (2, 3) => {
                let a_2d = a.as_rank2();
                let a_shape = a.shape();
                let b_shape = b.shape();

                let batch = b_shape[0];
                let _m = a_shape[0];
                let _k = a_shape[1];
                let n = b_shape[2];

                // Expand A to have a batch dimension
                let a_expanded = a_2d.unsqueeze_dim::<3>(0).repeat_dim(0, batch);
                let b_3d: Tensor<B, 3> = b.as_rank4().reshape([batch, b_shape[1], n]);

                let result = a_expanded.matmul(b_3d);
                DynTensor::from_rank3(result)
            }
            // 3D x 3D -> 3D (batched matmul)
            (3, 3) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let batch = a_shape[0].max(b_shape[0]);
                let m = a_shape[1];
                let k = a_shape[2];
                let n = b_shape[2];

                let a_3d: Tensor<B, 3> = a.as_rank4().reshape([a_shape[0], m, k]);
                let b_3d: Tensor<B, 3> = b.as_rank4().reshape([b_shape[0], b_shape[1], n]);

                // Handle broadcasting
                let a_3d = if a_shape[0] == 1 && batch > 1 {
                    a_3d.repeat_dim(0, batch)
                } else {
                    a_3d
                };
                let b_3d = if b_shape[0] == 1 && batch > 1 {
                    b_3d.repeat_dim(0, batch)
                } else {
                    b_3d
                };

                let result = a_3d.matmul(b_3d);
                DynTensor::from_rank3(result)
            }
            // 4D x 4D -> 4D (batched matmul with two batch dims)
            (4, 4) => {
                let a_4d = a.as_rank4();
                let b_4d = b.as_rank4();
                let result = a_4d.matmul(b_4d);
                DynTensor::from_rank4(result)
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported MatMul rank combination: {} x {}",
                    a_rank,
                    b_rank
                ));
            }
        };

        values.insert(output_name.clone(), output_dyn);
        Ok(())
    } else {
        Err(anyhow!("Not a MatMul node"))
    }
}

/// Gemm operator - General Matrix Multiplication
/// Y = alpha * A' * B' + beta * C
/// where A' is A (optionally transposed), B' is B (optionally transposed)
pub fn gemm<B: Backend>(node: &Node, values: &mut ValueStore<B>, device: &B::Device) -> Result<()> {
    if let Node::Gemm(n) = node {
        let output_name = &n.outputs[0].name;
        let config = &n.config;

        let a = get_input_tensor(&n.inputs[0], values, device)?;
        let b = get_input_tensor(&n.inputs[1], values, device)?;

        let mut a_2d = a.as_rank2();
        let mut b_2d = b.as_rank2();

        // Apply transpositions
        if config.trans_a != 0 {
            a_2d = a_2d.transpose();
        }
        if config.trans_b != 0 {
            b_2d = b_2d.transpose();
        }

        // Compute A * B
        let mut result = a_2d.matmul(b_2d);

        // Apply alpha scaling
        if (config.alpha - 1.0).abs() > 1e-6 {
            result = result.mul_scalar(config.alpha);
        }

        // Add C with beta scaling if C is provided
        if n.inputs.len() > 2 {
            let c = get_input_tensor(&n.inputs[2], values, device)?;
            let c_2d = c.as_rank2();

            if (config.beta - 0.0).abs() > 1e-6 {
                let c_scaled = if (config.beta - 1.0).abs() > 1e-6 {
                    c_2d.mul_scalar(config.beta)
                } else {
                    c_2d
                };
                result = result.add(c_scaled);
            }
        }

        values.insert(output_name.clone(), DynTensor::from_rank2(result));
        Ok(())
    } else {
        Err(anyhow!("Not a Gemm node"))
    }
}

pub fn linear<B: Backend>(
    node: &onnx_ir::ir::Node,
    values: &mut crate::runtime::value_store::ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    use anyhow::anyhow;
    use onnx_ir::ir::Node;

    if let Node::Linear(n) = node {
        let input_name = n.inputs[0].name.clone();
        let output_name = n.outputs[0].name.clone();
        let d_input = n.config.d_input;
        let d_output = n.config.d_output;
        let has_bias = n.config.bias;
        let transpose_weight = n.config.transpose_weight;

        println!(
            "  Linear node: input={}, output={}, dim=({}, {})",
            input_name, output_name, d_output, d_input
        );

        // Get input tensor from value store
        let input_dyn = values
            .get(&input_name)
            .ok_or_else(|| anyhow!("Input tensor '{}' not found", input_name))?;

        let input_tensor = input_dyn.as_rank2();
        println!("  Input tensor shape: {:?}", input_tensor.dims());

        // Get weights from Linear node arguments (indices 1 and 2)
        if n.inputs.len() < 2 {
            return Err(anyhow!("Linear node missing weights"));
        }

        // Extract weight from ONNX graph
        let weight_arg = &n.inputs[1];
        let weight_onnx_data = weight_arg
            .value()
            .ok_or_else(|| anyhow!("Could not get weight tensor data"))?;

        // Convert ONNX TensorData to f32 vec
        let weight_slice = weight_onnx_data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("Could not convert weight to f32: {:?}", e))?;
        let weight_vec = weight_slice.to_vec();

        // For Linear with transpose_weight=true in ONNX, weights are stored as [out_features, in_features]
        // but for matmul we need them as [in_features, out_features] if we're doing input @ weight
        let weight: Tensor<B, 2> = if transpose_weight {
            // ONNX weights: [out_features, in_features] = [2, 4]
            // Create with that shape and transpose for matmul
            let weight_tensor_data = TensorData::new(weight_vec, [d_output, d_input]);
            Tensor::from_data(weight_tensor_data, device).transpose()
        } else {
            let weight_tensor_data = TensorData::new(weight_vec, [d_input, d_output]);
            Tensor::from_data(weight_tensor_data, device)
        };

        println!("  Weight shape: {:?}", weight.dims());

        let result = if has_bias && n.inputs.len() > 2 {
            let bias_arg = &n.inputs[2];
            let bias_onnx_data = bias_arg
                .value()
                .ok_or_else(|| anyhow!("Could not get bias tensor data"))?;

            let bias_slice = bias_onnx_data
                .as_slice::<f32>()
                .map_err(|e| anyhow!("Could not convert bias to f32: {:?}", e))?;
            let bias_vec = bias_slice.to_vec();

            let bias_tensor_data = TensorData::new(bias_vec, [d_output]);
            let bias: Tensor<B, 1> = Tensor::from_data(bias_tensor_data, device);

            println!("  Bias shape: {:?}", bias.dims());

            Some(bias)
        } else {
            None
        };

        let result = if let Some(bias) = result {
            // output = input @ weight + bias
            input_tensor
                .matmul(weight.clone())
                .add(bias.unsqueeze_dim(0))
        } else {
            // output = input @ weight
            input_tensor.matmul(weight.clone())
        };

        // Store output
        println!("  Linear node output: {}", output_name);
        let dyn_output = DynTensor::from_rank2(result);
        values.insert(output_name, dyn_output);

        Ok(())
    } else {
        Err(anyhow!("Not a Linear node"))
    }
}
