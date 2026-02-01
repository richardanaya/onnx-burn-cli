#!/usr/bin/env python3
"""Generate simple ONNX test models for operator testing"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_unary_op_model(op_name, op_type, input_shape):
    """Create a simple unary op ONNX model"""
    # Create graph
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)

    # Create node
    node = helper.make_node(op_type, ['X'], ['Y'], name='test_node')

    # Create graph
    graph = helper.make_graph([node], f'{op_name}_test', [X], [Y])

    # Create model
    model = helper.make_model(graph, producer_name='nnx-tests')
    model.opset_import[0].version = 19

    return model

def create_binary_op_model(op_name, op_type, input_shape):
    """Create a simple binary op ONNX model"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, input_shape)

    node = helper.make_node(op_type, ['X', 'Y'], ['Z'], name='test_node')

    graph = helper.make_graph([node], f'{op_name}_test', [X, Y], [Z])
    model = helper.make_model(graph, producer_name='nnx-tests')
    model.opset_import[0].version = 19

    return model

def main():
    import os
    os.makedirs('test_data/operator_tests', exist_ok=True)

    # Unary math ops
    for op in ['Abs', 'Neg', 'Sqrt', 'Exp', 'Log', 'Ceil', 'Floor', 'Round', 'Sign',
               'Reciprocal', 'Sin', 'Cos', 'Tan', 'Sinh', 'Cosh']:
        model = create_unary_op_model(op, op, [2, 3, 4])
        onnx.save(model, f'test_data/operator_tests/{op.lower()}_test.onnx')

    # Activation functions
    for op in ['Relu', 'Sigmoid', 'Tanh', 'Gelu']:
        model = create_unary_op_model(op, op, [2, 3, 4])
        onnx.save(model, f'test_data/operator_tests/{op.lower()}_test.onnx')

    # Shape ops
    for op in ['Flatten', 'Reshape', 'Squeeze', 'Unsqueeze', 'Transpose', 'Concat', 'Split', 'Slice', 'Expand', 'Tile']:
        # These are more complex, skip for now
        pass

    print("Generated test models in test_data/operator_tests/")

if __name__ == '__main__':
    main()