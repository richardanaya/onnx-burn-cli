## ADDED Requirements

### Requirement: Parse ONNX model files
The system SHALL parse ONNX model files using the onnx-ir crate.

#### Scenario: Valid ONNX file
- **WHEN** a valid .onnx file path is provided
- **THEN** system parses the file and returns the graph structure

#### Scenario: Invalid ONNX file
- **WHEN** an invalid or corrupted .onnx file is provided
- **THEN** system returns an error describing the parse failure

### Requirement: Extract model inputs
The system SHALL extract input tensor specifications from the model.

#### Scenario: Display inputs
- **WHEN** model is parsed
- **THEN** system displays each input's name, data type, and shape

### Requirement: Extract model outputs
The system SHALL extract output tensor specifications from the model.

#### Scenario: Display outputs
- **WHEN** model is parsed
- **THEN** system displays each output's name, data type, and shape

### Requirement: List operators used
The system SHALL list all ONNX operators used in the model.

#### Scenario: Display operator list
- **WHEN** model is parsed
- **THEN** system displays a deduplicated list of operator types used (e.g., Conv, Relu, MatMul)

### Requirement: Indicate operator support status
The system SHALL indicate which operators are supported by the runtime.

#### Scenario: Show supported vs unsupported
- **WHEN** operator list is displayed
- **THEN** system marks each operator as supported or unsupported
