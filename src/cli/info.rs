use anyhow::Result;
use onnx_ir::{
    ir::{ArgType, DType, Node, TensorType as IrTensorType},
    OnnxGraphBuilder,
};
use std::collections::HashSet;
use std::path::Path;

/// Format an ArgType for display
fn format_arg_type(ty: &ArgType) -> String {
    match ty {
        ArgType::Tensor(tensor_type) => format!("{:?}", tensor_type),
        ArgType::Scalar(dtype) => format!("{:?}", dtype),
        ArgType::Shape(rank) => format!("Shape(rank={})", rank),
    }
}

/// Helper to format tensor type for display
fn format_tensor_type(ty: &ArgType) -> String {
    format_arg_type(ty)
}

pub fn load_model<P: AsRef<Path>>(path: P) -> Result<onnx_ir::ir::OnnxGraph> {
    let graph = OnnxGraphBuilder::new().parse_file(path)?;
    Ok(graph)
}

/// Get the display name for a Node variant
fn get_node_op_name(node: &Node) -> String {
    let variant_name = format!("{:?}", node);
    variant_name
        .split('(')
        .next()
        .unwrap_or("Unknown")
        .to_string()
}

/// Check if an operator is supported
fn is_supported_op(name: &str) -> bool {
    const SUPPORTED_OPS: &[&str] = &[
        // Arithmetic operations
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Max",
        "Min",
        "Mod",
        "Sum",
        "Mean",
        // Unary math operations
        "Abs",
        "Neg",
        "Sqrt",
        "Exp",
        "Log",
        "Ceil",
        "Floor",
        "Round",
        "Sign",
        "Reciprocal",
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Erf",
        // Activation functions
        "Relu",
        "Sigmoid",
        "Softmax",
        "Tanh",
        "Gelu",
        "LeakyRelu",
        "HardSigmoid",
        "HardSwish",
        "PRelu",
        "LogSoftmax",
        "Identity",
        "Dropout",
        "Clip",
        // Convolution and pooling
        "Conv",
        "Conv1d",
        "Conv2d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "MaxPool",
        "MaxPool1d",
        "MaxPool2d",
        "AveragePool",
        "AveragePool1d",
        "AveragePool2d",
        "GlobalAveragePool",
        "BatchNormalization",
        // Linear algebra
        "MatMul",
        "Gemm",
        "Linear",
        // Shape operations
        "Flatten",
        "Reshape",
        "Shape",
        "Squeeze",
        "Unsqueeze",
        "Transpose",
        "Concat",
        "Split",
        "Slice",
        "Expand",
        "Tile",
        "Pad",
        "Size",
        // Reduction operations
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ArgMax",
        "ArgMin",
        // Comparison operations
        "Equal",
        "Greater",
        "Less",
        "GreaterOrEqual",
        "LessOrEqual",
        "IsInf",
        "IsNaN",
        // Normalization operations
        "LayerNormalization",
        "InstanceNormalization",
        "GroupNormalization",
        // Advanced operations
        "Gather",
        "GatherElements",
        "Where",
        "TopK",
        "CumSum",
        // Type conversion
        "Cast",
        "ConstantOfShape",
        // Spatial transform
        "Resize",
        "DepthToSpace",
        "SpaceToDepth",
        // Advanced operations - additional
        "OneHot",
        // Other
        "Constant",
    ];

    SUPPORTED_OPS.contains(&name)
}

pub fn info_impl(model: &str) -> Result<()> {
    let graph = load_model(model)?;

    // Display inputs
    println!("Inputs:");
    for input in &graph.inputs {
        println!("  - {}: {}", input.name, format_tensor_type(&input.ty));
    }

    // Display outputs
    println!("\nOutputs:");
    for output in &graph.outputs {
        println!("  - {}: {}", output.name, format_tensor_type(&output.ty));
    }

    // Display operators
    println!("\nOperators:");
    let mut ops: HashSet<String> = HashSet::new();
    for node in &graph.nodes {
        ops.insert(get_node_op_name(node));
    }

    let mut ops_vec: Vec<_> = ops.into_iter().collect();
    ops_vec.sort();

    ops_vec.iter().for_each(|op| {
        let support = if is_supported_op(op) {
            "[✓]"
        } else {
            "[✗]"
        };
        println!("  {} {}", support, op);
    });

    // Total stats
    println!("\nModel Statistics:");
    println!("  Total nodes: {}", graph.nodes.len());
    println!("  Unique operators: {}", ops_vec.len());

    Ok(())
}
