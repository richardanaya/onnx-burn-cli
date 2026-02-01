use crate::cli::info::load_model;
use crate::input::image;
use crate::runtime::executor;
use anyhow::{Result, anyhow};
use burn_wgpu::Wgpu;
use burn::{prelude::Backend, tensor::Tensor, tensor::TensorData};
use std::collections::HashMap;

type OurBackend = Wgpu;

pub async fn infer_impl(
    model: &str,
    input_path: Option<&str>,
    labels_path: Option<&str>,
    top_k: usize,
) -> Result<()> {
    println!("Running inference on model: {}", model);

    // Load model
    let graph = load_model(model)?;
    println!("Loaded model with {} nodes", graph.nodes.len());

    // Check model inputs
    let input_info = graph
        .inputs
        .first()
        .ok_or_else(|| anyhow!("Model has no inputs"))?;
    let input_name = input_info.name.clone();
    println!("Model expects input: {}", input_name);
    
    // Extract input shape from tensor type
    let (input_rank, input_shape) = if let onnx_ir::ir::ArgType::Tensor(tensor_type) = &input_info.ty {
        (tensor_type.rank as usize, tensor_type.static_shape.clone())
    } else {
        return Err(anyhow!("Input is not a tensor type"));
    };
    println!("Input rank: {}, shape: {:?}", input_rank, input_shape);

    // Check model outputs
    let output_info = graph
        .outputs
        .first()
        .ok_or_else(|| anyhow!("Model has no outputs"))?;
    let output_name = output_info.name.clone();
    println!("Model produces output: {}", output_name);

    // Load labels file if provided
    let mut labels: Option<Vec<String>> = None;
    if let Some(path) = labels_path {
        println!("Loading labels from: {}", path);
        labels = Some(image::load_labels(path)?);
        println!("Loaded {} class labels", labels.as_ref().unwrap().len());
    }

    // Initialize GPU device
    let device = burn_wgpu::WgpuDevice::default();
    println!("Using GPU device for inference");

    // Validate model has operators we support
    let unsupported_ops: Vec<String> = graph
        .nodes
        .iter()
        .filter(|node| {
            let op_name = format!("{:?}", node).split('(').next().unwrap_or("Unknown").to_string();
            !is_supported_op(&op_name)
        })
        .map(|node| format!("{}", node.name()))
        .collect();

    if !unsupported_ops.is_empty() {
        return Err(anyhow::anyhow!(
            "Model contains unsupported operators: {}. Use 'nnx info' to see supported operators.",
            unsupported_ops.join(", ")
        ));
    }

    // Load and process input
    let input_dyn = if let Some(img_path) = input_path {
        println!("Loading image: {}", img_path);
        
        // Use static shape if available, otherwise error
        match (input_rank, &input_shape) {
            (4, Some(shape)) => {
                // Image model: [batch, channels, height, width]
                let [_batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
                let pixels = image::load_and_preprocess(img_path, width, height)?;
                // For grayscale models, take only first channel
                let pixels: Vec<f32> = if channels == 1 {
                    pixels.chunks(height * width).next().unwrap_or(&[]).to_vec()
                } else {
                    pixels
                };
                let input_data = TensorData::new(pixels, [1, channels, height, width]);
                let input_tensor = Tensor::<OurBackend, 4>::from_data(input_data, &device);
                crate::runtime::tensor::DynTensor::from_rank4(input_tensor)
            }
            (2, Some(shape)) => {
                // Flat model: [batch, features]
                let features = shape[1];
                let side = (features as f64).sqrt() as usize;
                let pixels = if img_path.ends_with(".jpg") || img_path.ends_with(".png") {
                    let img_pixels = image::load_and_preprocess(img_path, side, side)?;
                    img_pixels.chunks(side * side).next().unwrap_or(&[]).to_vec()
                } else {
                    vec![1.0f32; features]
                };
                let input_data = TensorData::new(pixels, [1, features]);
                let input_tensor = Tensor::<OurBackend, 2>::from_data(input_data, &device);
                crate::runtime::tensor::DynTensor::from_rank2(input_tensor)
            }
            (_, None) => {
                return Err(anyhow!(
                    "Model input shape is not statically known. \
                     Please use a model with concrete input shapes. \
                     You can fix this by making the batch dimension concrete in the ONNX model."
                ));
            }
            _ => return Err(anyhow!("Unsupported input rank: {}", input_rank)),
        }
    } else {
        // Create dummy input tensor
        println!("No input provided, using dummy data");
        match (input_rank, &input_shape) {
            (4, Some(shape)) => {
                let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
                let input_data = TensorData::new(vec![0.5f32; batch * channels * height * width], [batch, channels, height, width]);
                let input_tensor = Tensor::<OurBackend, 4>::from_data(input_data, &device);
                crate::runtime::tensor::DynTensor::from_rank4(input_tensor)
            }
            (2, Some(shape)) => {
                let [batch, features] = [shape[0], shape[1]];
                let input_data = TensorData::new(vec![0.5f32; batch * features], [batch, features]);
                let input_tensor = Tensor::<OurBackend, 2>::from_data(input_data, &device);
                crate::runtime::tensor::DynTensor::from_rank2(input_tensor)
            }
            (_, None) => {
                return Err(anyhow!(
                    "Model input shape is not statically known. \
                     Please use a model with concrete input shapes."
                ));
            }
            _ => return Err(anyhow!("Unsupported input rank: {}", input_rank)),
        }
    };

    // Convert to ValueStore
    let mut inputs = HashMap::new();
    inputs.insert(input_name, input_dyn);

    // Execute the graph
    println!("\n[+] Executing inference...");
    let output_store = executor::execute_graph(&graph, inputs, &device)?;

    // Get output
    let output = output_store
        .get(&output_name)
        .ok_or_else(|| anyhow!("Output not found in value store"))?;

    println!("[+] Inference complete\n");

    // Extract output values for display
    let output_tensor_2d = output.as_rank2();
    let output_data: burn::tensor::TensorData = output_tensor_2d.into_data_async().await?;
    let output_values: &[f32] = output_data.as_slice()?;

    // Display predictions
    display_predictions(output_values, labels.as_deref(), top_k)?;

    Ok(())
}

fn display_predictions(
    values: &[f32],
    labels: Option<&[String]>,
    top_k: usize,
) -> Result<()> {
    if top_k > 0 {
        // Top-k display
        let num_to_show = top_k.min(values.len());
        println!("Top-{} predictions:", num_to_show);

        // Collect and sort indices by score
        let mut with_indices: Vec<(usize, &f32)> = values.iter().enumerate().collect();
        with_indices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..num_to_show {
            let (idx, &score) = with_indices[i];
            if let Some(label_names) = labels {
                let label = label_names.get(idx).map(|s| s.as_str()).unwrap_or("unknown");
                println!("  {}. {} ({:.4})", i + 1, label, score);
            } else {
                println!("  {}. Class {} ({:.4})", i + 1, idx, score);
            }
        }
    } else {
        // Full output display
        println!("Output tensor: {} values", values.len());
        println!("\n[+] Output values:");
        for (i, &val) in values.iter().enumerate() {
            if let Some(label_names) = labels {
                let label = label_names.get(i).map(|s| s.as_str()).unwrap_or("unknown");
                println!("  {} = {:.6} ({})", label, val, i);
            } else {
                println!("  Output[{}] = {:.6}", i, val);
            }
        }
    }

    Ok(())
}

fn is_supported_op(name: &str) -> bool {
    const SUPPORTED_OPS: &[&str] = &[
        // Arithmetic operations
        "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Mod", "Sum", "Mean",
        // Unary math operations
        "Abs", "Neg", "Sqrt", "Exp", "Log", "Ceil", "Floor", "Round", "Sign",
        "Reciprocal", "Sin", "Cos", "Tan", "Sinh", "Cosh", "Erf",
        // Activation functions
        "Relu", "Sigmoid", "Softmax", "Tanh", "Gelu", "LeakyRelu", "HardSigmoid",
        "HardSwish", "PRelu", "LogSoftmax", "Identity", "Dropout", "Clip",
        // Convolution and pooling
        "Conv", "Conv1d", "Conv2d", "ConvTranspose1d", "ConvTranspose2d",
        "MaxPool", "MaxPool1d", "MaxPool2d", "AveragePool", "AveragePool1d", "AveragePool2d",
        "GlobalAveragePool", "BatchNormalization",
        // Linear algebra
        "MatMul", "Gemm", "Linear",
        // Shape operations
        "Flatten", "Reshape", "Shape", "Squeeze", "Unsqueeze", "Transpose", "Concat",
        "Split", "Slice", "Expand", "Tile", "Pad", "Size",
        // Reduction operations
        "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
        "ArgMax", "ArgMin",
        // Comparison operations
        "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "IsInf", "IsNaN",
        // Normalization operations
        "LayerNormalization", "InstanceNormalization", "GroupNormalization",
        // Advanced operations
        "Gather", "GatherElements", "Where", "TopK", "CumSum",
        // Type conversion
        "Cast", "ConstantOfShape",
        // Spatial transform
        "Resize", "DepthToSpace", "SpaceToDepth",
        // Advanced operations - additional
        "OneHot",
        // Other
        "Constant",
    ];
    SUPPORTED_OPS.contains(&name)
}