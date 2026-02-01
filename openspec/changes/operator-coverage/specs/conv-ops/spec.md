## ADDED Requirements

### Requirement: Conv1d and Conv3d operations
The runtime SHALL support 1D and 3D convolutions in addition to existing Conv2d.

#### Scenario: Conv1d forward pass
- **WHEN** executor encounters Conv1d node with input shape [1, 64, 100], weight shape [128, 64, 3], stride=1, padding=1
- **THEN** output tensor SHALL have shape [1, 128, 100]

#### Scenario: Conv3d forward pass
- **WHEN** executor encounters Conv3d node with input shape [1, 3, 8, 224, 224], weight shape [64, 3, 3, 7, 7]
- **THEN** output tensor SHALL have appropriate 5D output shape based on stride/padding

### Requirement: Transposed convolution operations
The runtime SHALL support transposed (deconvolution) operations for upsampling.

- `ConvTranspose1d`: 1D transposed convolution
- `ConvTranspose2d`: 2D transposed convolution
- `ConvTranspose3d`: 3D transposed convolution

#### Scenario: ConvTranspose2d upsampling
- **WHEN** executor encounters ConvTranspose2d node with input shape [1, 512, 7, 7], weight shape [512, 256, 4, 4], stride=2, padding=1
- **THEN** output tensor SHALL have shape [1, 256, 14, 14] (2x upsampling)

#### Scenario: ConvTranspose1d
- **WHEN** executor encounters ConvTranspose1d node with input shape [1, 64, 50], stride=2
- **THEN** output tensor SHALL have approximately doubled length along the sequence dimension
