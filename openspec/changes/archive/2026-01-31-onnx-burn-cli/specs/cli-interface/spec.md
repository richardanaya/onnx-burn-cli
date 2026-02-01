## ADDED Requirements

### Requirement: CLI provides devices subcommand
The CLI SHALL provide a `devices` subcommand that lists available GPU devices.

#### Scenario: List devices
- **WHEN** user runs `nnx devices`
- **THEN** system displays available GPU adapters with their names and backend info

### Requirement: CLI provides info subcommand
The CLI SHALL provide an `info` subcommand that displays ONNX model metadata.

#### Scenario: Show model info
- **WHEN** user runs `nnx info <model.onnx>`
- **THEN** system displays model inputs, outputs, and operators used

#### Scenario: Model file not found
- **WHEN** user runs `nnx info` with a non-existent file path
- **THEN** system displays an error message indicating the file was not found

### Requirement: CLI provides infer subcommand
The CLI SHALL provide an `infer` subcommand that runs inference on an ONNX model.

#### Scenario: Basic inference with image input
- **WHEN** user runs `nnx infer <model.onnx> -i <image.png>`
- **THEN** system loads the model, processes the image, runs inference, and displays output

#### Scenario: Inference with labels file
- **WHEN** user runs `nnx infer <model.onnx> -i <image.png> --labels <labels.txt>`
- **THEN** system displays class names from labels file alongside prediction scores

#### Scenario: Top-k results
- **WHEN** user runs `nnx infer <model.onnx> -i <image.png> --top 5`
- **THEN** system displays only the top 5 predictions sorted by score

### Requirement: CLI provides help and version
The CLI SHALL provide `--help` and `--version` flags.

#### Scenario: Display help
- **WHEN** user runs `nnx --help`
- **THEN** system displays usage information and available subcommands

#### Scenario: Display version
- **WHEN** user runs `nnx --version`
- **THEN** system displays the version number
