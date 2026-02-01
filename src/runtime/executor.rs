use crate::runtime::value_store::ValueStore;
use crate::runtime::weight_loader::load_weights;
use anyhow::Result;
use burn::prelude::Backend;
use std::collections::HashMap;

/// Execute a single ONNX node using the dispatcher
pub fn execute_node<B: Backend>(
    node: &onnx_ir::ir::Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    crate::ops::dispatch(node, values, device)?;
    Ok(())
}

/// Execute the entire ONNX graph in topological order
pub fn execute_graph<B: Backend>(
    graph: &onnx_ir::ir::OnnxGraph,
    inputs: HashMap<String, crate::runtime::tensor::DynTensor<B>>,
    device: &B::Device,
) -> Result<ValueStore<B>> {
    let mut values = ValueStore::new();

    // Load weights from ONNX initializers into ValueStore
    load_weights(graph, &mut values, device)?;

    // Add input tensors
    values.extend(inputs);

    // Execute all nodes in topological order
    for node in &graph.nodes {
        // Skip constant nodes (weights already loaded)
        if !matches!(node, onnx_ir::ir::Node::Constant(_)) {
            execute_node(node, &mut values, device)?;
        }
    }

    Ok(values)
}
