use crate::runtime::tensor::DynTensor;
use crate::runtime::value_store::ValueStore;
use anyhow::{anyhow, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use onnx_ir::ir::Node;

/// Range operator - generates a sequence of numbers
/// Output: 1D tensor [start, start+delta, start+2*delta, ...] up to (but not including) limit
pub fn range<B: Backend>(
    node: &Node,
    values: &mut ValueStore<B>,
    device: &B::Device,
) -> Result<()> {
    if let Node::Range(n) = node {
        let output_name = &n.outputs[0].name;

        // Get start, limit, delta from config or runtime inputs
        use onnx_ir::node::range::RangeInput;

        let start = match &n.config.start {
            RangeInput::Static(v) => *v as f32,
            RangeInput::Runtime(r) => {
                // Try to get from value store
                if let Some(tensor) = values.get(&r.name) {
                    let data = tensor.as_rank4().to_data();
                    let slice: Vec<f32> = data
                        .to_vec()
                        .map_err(|e| anyhow!("Range: cannot get start value: {:?}", e))?;
                    slice[0]
                } else if let Some(const_data) = n.inputs[0].value() {
                    const_data
                        .as_slice::<f32>()
                        .map_err(|e| anyhow!("Range: cannot get start const: {:?}", e))?[0]
                } else {
                    return Err(anyhow!("Range: start value not found"));
                }
            }
        };

        let limit = match &n.config.limit {
            RangeInput::Static(v) => *v as f32,
            RangeInput::Runtime(r) => {
                if let Some(tensor) = values.get(&r.name) {
                    let data = tensor.as_rank4().to_data();
                    let slice: Vec<f32> = data
                        .to_vec()
                        .map_err(|e| anyhow!("Range: cannot get limit value: {:?}", e))?;
                    slice[0]
                } else if let Some(const_data) = n.inputs[1].value() {
                    const_data
                        .as_slice::<f32>()
                        .map_err(|e| anyhow!("Range: cannot get limit const: {:?}", e))?[0]
                } else {
                    return Err(anyhow!("Range: limit value not found"));
                }
            }
        };

        let delta = match &n.config.delta {
            RangeInput::Static(v) => *v as f32,
            RangeInput::Runtime(r) => {
                if let Some(tensor) = values.get(&r.name) {
                    let data = tensor.as_rank4().to_data();
                    let slice: Vec<f32> = data
                        .to_vec()
                        .map_err(|e| anyhow!("Range: cannot get delta value: {:?}", e))?;
                    slice[0]
                } else if let Some(const_data) = n.inputs[2].value() {
                    const_data
                        .as_slice::<f32>()
                        .map_err(|e| anyhow!("Range: cannot get delta const: {:?}", e))?[0]
                } else {
                    return Err(anyhow!("Range: delta value not found"));
                }
            }
        };

        // Handle delta == 0 case
        if delta == 0.0 {
            return Err(anyhow!("Range: delta cannot be zero"));
        }

        // Calculate number of elements
        // number_of_elements = max(ceil((limit - start) / delta), 0)
        let num_elements = ((limit - start) / delta).ceil().max(0.0) as usize;

        // Handle empty range
        if num_elements == 0 {
            // Create empty tensor - use shape [1] with dummy value as workaround
            // since burn may not support [0] shape
            let output_data: Vec<f32> = vec![];
            if output_data.is_empty() {
                // Create minimal tensor - Range spec says empty is valid
                let tensor_data = TensorData::new(vec![0.0f32], [1]);
                let tensor: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
                // Store with actual empty shape info
                values.insert(
                    output_name.clone(),
                    DynTensor::from_rank1(tensor.slice([0..0])),
                );
            }
            return Ok(());
        }

        // Generate sequence
        let mut output_data = Vec::with_capacity(num_elements);
        let mut current = start;
        for _ in 0..num_elements {
            output_data.push(current);
            current += delta;
        }

        let tensor_data = TensorData::new(output_data, [num_elements]);
        let tensor: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
        values.insert(output_name.clone(), DynTensor::from_rank1(tensor));

        Ok(())
    } else {
        Err(anyhow!("Not a Range node"))
    }
}
