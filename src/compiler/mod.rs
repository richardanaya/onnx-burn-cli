use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use std::collections::{HashMap, HashSet};

/// Represents a fused operation in the execution plan
#[derive(Clone, Debug)]
pub enum CompiledOp {
    /// Single operator execution
    Single {
        node_index: usize,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    /// Convolution + Activation fusion
    ConvActFusion {
        conv_node_idx: usize,
        act_node_idx: usize,
        inputs: Vec<String>,
        outputs: Vec<String>,
        act_type: String,
    },
    /// Linear + Activation fusion
    LinearActFusion {
        linear_node_idx: usize,
        act_node_idx: usize,
        inputs: Vec<String>,
        outputs: Vec<String>,
        act_type: String,
    },
    /// Conv + BatchNorm + Activation fusion
    ConvBNActFusion {
        conv_node_idx: usize,
        bn_node_idx: usize,
        act_node_idx: usize,
        inputs: Vec<String>,
        outputs: Vec<String>,
        act_type: String,
    },
}

/// Execution plan for the compiled graph
pub struct ExecutionPlan {
    pub ops: Vec<CompiledOp>,
    pub tensor_shapes: HashMap<String, Vec<usize>>,
}

/// Node information for compilation
struct NodeInfo {
    idx: usize,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

/// Compiler for ONNX graph optimization
pub struct GraphCompiler;

impl GraphCompiler {
    /// Compile an ONNX graph into an optimized execution plan
    pub fn compile(graph: &onnx_ir::ir::OnnxGraph) -> Result<ExecutionPlan> {
        let mut node_info = Self::build_node_info(graph);
        let mut ops = Vec::new();
        let mut processed = HashSet::new();

        for node_idx in 0..graph.nodes.len() {
            if processed.contains(&node_idx) {
                continue;
            }

            let node = &graph.nodes[node_idx];
            if matches!(node, onnx_ir::ir::Node::Constant(_)) {
                processed.insert(node_idx);
                continue;
            }

            let info = &node_info[node_idx];

            // Try fusion patterns first
            if let Some(fused) =
                Self::try_fusion(&node_idx, &node_info, &graph.nodes, &mut processed)
            {
                ops.push(fused);
            } else {
                // Single operator execution
                ops.push(CompiledOp::Single {
                    node_index: node_idx,
                    op_type: info.op_type.clone(),
                    inputs: info.inputs.clone(),
                    outputs: information.outputs.clone(),
                });
                processed.insert(node_idx);
            }
        }

        // Build tensor shape information
        let mut tensor_shapes = HashMap::new();

        // Add input shapes
        for input in &graph.inputs {
            if let onnx_ir::ir::ArgType::Tensor(tensor_type) = &input.ty {
                if let Some(shape) = &tensor_type.static_shape {
                    tensor_shapes.insert(input.name.clone(), shape.clone());
                }
            }
        }

        // Track shapes through the graph
        for op in &ops {
            match op {
                CompiledOp::Single { outputs, .. } => {
                    for output in outputs {
                        if !tensor_shapes.contains_key(output) {
                            // For unsupported nodes, we'd need shape inference
                            // For now, mark as unknown
                            tensor_shapes.insert(output.clone(), vec![]);
                        }
                    }
                }
                CompiledOp::ConvActFusion { outputs, .. } => {
                    for output in outputs {
                        if !tensor_shapes.contains_key(output) {
                            tensor_shapes.insert(output.clone(), vec![]);
                        }
                    }
                }
                CompiledOp::LinearActFusion { outputs, .. } => {
                    for output in outputs {
                        if !tensor_shapes.contains_key(output) {
                            tensor_shapes.insert(output.clone(), vec![]);
                        }
                    }
                }
                CompiledOp::ConvBNActFusion { outputs, .. } => {
                    for output in outputs {
                        if !tensor_shapes.contains_key(output) {
                            tensor_shapes.insert(output.clone(), vec![]);
                        }
                    }
                }
            }
        }

        Ok(ExecutionPlan { ops, tensor_shapes })
    }

    /// Build node information for compilation
    fn build_node_info(graph: &onnx_ir::ir::OnnxGraph) -> Vec<NodeInfo> {
        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| {
                let op_type = format!("{:?}", node)
                    .split('(')
                    .next()
                    .unwrap_or("Unknown")
                    .to_string();
                let (inputs, outputs) = Self::get_node_io(node);

                NodeInfo {
                    idx,
                    op_type,
                    inputs,
                    outputs,
                }
            })
            .collect()
    }

    /// Get input and output names from a node
    fn get_node_io(node: &onnx_ir::ir::Node) -> (Vec<String>, Vec<String>) {
        use onnx_ir::ir::Node;

        match node {
            Node::Add(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Conv2d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::MatMul(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Gemm(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Linear(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Relu(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Sigmoid(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Softmax(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Tanh(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::BatchNormalization(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Conv1d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ConvTranspose2d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::MaxPool2d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::MaxPool1d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::AveragePool2d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::AveragePool1d(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::MatMul(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Flatten(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Reshape(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Shape(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Concat(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Transpose(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Squeeze(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Unsqueeze(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Gelu(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::LeakyRelu(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::HardSigmoid(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::HardSwish(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::PRelu(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::LogSoftmax(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Identity(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Dropout(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Clip(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::LayerNormalization(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::InstanceNormalization(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::GroupNormalization(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::GlobalAveragePool(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Size(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ReduceSum(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ReduceMean(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ReduceMax(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ReduceMin(n) => (n.inputs.iter().map(|i| i.name.clone()).collect()),
            Node::ReduceProd(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ArgMax(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::ArgMin(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Equal(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Greater(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Less(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::GreaterOrEqual(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::LessOrEqual(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::IsInf(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::IsNaN(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Pow(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Max(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Min(n) => (n.inputs.iter().map(|i| i.name.clone()).collect()),
            Node::Mod(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Sum(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Mean(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Gather(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::GatherElements(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Where(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::TopK(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::CumSum(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Split(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Slice(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Expand(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Tile(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Pad(n) => (
                n.inputs.iter().map(|i| i.name.clone()).collect(),
                n.outputs.iter().map(|i| i.name.clone()).collect(),
            ),
            Node::Constant(_) => (vec![], vec![]),
            _ => (vec![], vec![]),
        }
    }

    /// Try to fuse operators starting from a node
    fn try_fusion(
        node_idx: &usize,
        node_info: &[NodeInfo],
        nodes: &[onnx_ir::ir::Node],
        processed: &mut HashSet<usize>,
    ) -> Option<CompiledOp> {
        let current_info = &node_info[*node_idx];

        // Check if current node can be fused with next node
        if current_info.outputs.len() != 1 {
            return None;
        }

        let conv_output = &current_info.outputs[0];

        // Find all nodes that take this output as input
        let consumers: Vec<&NodeInfo> = node_info
            .iter()
            .filter(|info| info.inputs.contains(conv_output))
            .collect();

        if consumers.len() != 1 {
            return None;
        }

        let consumer_idx = consumers[0].idx;
        let consumer_info = &node_info[consumer_idx];
        let consumer_node = &nodes[consumer_idx];

        let is_activation = matches!(consumer_node, onnx_ir::ir::Node::Relu(_));

        if is_activation {
            // Check for Conv+Act fusion
            if Self::is_conv_op(&current_info.op_type) {
                processed.insert(*node_idx);
                processed.insert(consumer_idx);

                return Some(CompiledOp::ConvActFusion {
                    conv_node_idx: *node_idx,
                    act_node_idx: consumer_idx,
                    inputs: current_info.inputs.clone(),
                    outputs: consumer_info.outputs.clone(),
                    act_type: consumer_info.op_type.clone(),
                });
            }

            // Check for Linear+Act fusion
            if Self::is_linear_op(&current_info.op_type) {
                processed.insert(*node_idx);
                processed.insert(consumer_idx);

                return Some(CompiledOp::LinearActFusion {
                    linear_node_idx: *node_idx,
                    act_node_idx: consumer_idx,
                    inputs: current_info.inputs.clone(),
                    outputs: consumer_info.outputs.clone(),
                    act_type: consumer_info.op_type.clone(),
                });
            }
        }

        // Check for Conv+BN+Act fusion
        if let Some(bn_idx) = Self::find_batchnorm_consumer(conv_output, node_info) {
            let bn_info = &node_info[bn_idx];
            if bn_info.outputs.len() != 1 {
                return None;
            }
            let bn_output = &bn_info.outputs[0];

            if let Some(act_idx) = Self::find_activation_consumer(bn_output, node_info, nodes) {
                let act_info = &node_info[act_idx];

                processed.insert(*node_idx);
                processed.insert(bn_idx);
                processed.insert(act_idx);

                return Some(CompiledOp::ConvBNActFusion {
                    conv_node_idx: *node_idx,
                    bn_node_idx: bn_idx,
                    act_node_idx: act_idx,
                    inputs: current_info.inputs.clone(),
                    outputs: act_info.outputs.clone(),
                    act_type: act_info.op_type.clone(),
                });
            }
        }

        None
    }

    /// Check if operation is a convolution
    fn is_conv_op(op_type: &str) -> bool {
        matches!(op_type, "Conv2d" | "Conv1d" | "ConvTranspose2d" | "Conv3d")
    }

    /// Check if operation is linear
    fn is_linear_op(op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm" | "Linear")
    }

    /// Find BatchNormalization node that consumes given output
    fn find_batchnorm_consumer(output: &str, node_info: &[NodeInfo]) -> Option<usize> {
        node_info
            .iter()
            .find(|info| info.inputs.contains(output) && info.op_type == "BatchNormalization")
            .map(|info| info.idx)
    }

    /// Find activation node that consumes given output
    fn find_activation_consumer(
        output: &str,
        node_info: &[NodeInfo],
        nodes: &[onnx_ir::ir::Node],
    ) -> Option<usize> {
        node_info
            .iter()
            .find(|info| info.inputs.contains(output) && Self::is_activation_op(&nodes[info.idx]))
            .map(|info| info.idx)
    }

    /// Check if node is an activation function
    fn is_activation_op(node: &onnx_ir::ir::Node) -> bool {
        matches!(
            node,
            onnx_ir::ir::Node::Relu(_)
                | onnx_ir::ir::Node::Sigmoid(_)
                | onnx_ir::ir::Node::Softmax(_)
                | onnx_ir::ir::Node::Tanh(_)
                | onnx_ir::ir::Node::Gelu(_)
        )
    }
}
