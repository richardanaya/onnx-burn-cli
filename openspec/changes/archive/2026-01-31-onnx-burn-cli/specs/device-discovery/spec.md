## ADDED Requirements

### Requirement: Enumerate GPU devices via wgpu
The system SHALL enumerate available GPU devices using the wgpu backend.

#### Scenario: List available adapters
- **WHEN** device discovery is requested
- **THEN** system queries wgpu for all available adapters and returns their info

#### Scenario: No GPU available
- **WHEN** device discovery is requested and no GPU is available
- **THEN** system indicates no compatible GPU devices were found

### Requirement: Display device information
The system SHALL display meaningful device information including adapter name and backend type.

#### Scenario: Show device details
- **WHEN** a device is discovered
- **THEN** system displays the adapter name (e.g., "NVIDIA GeForce RTX 3080") and backend (e.g., "Vulkan")

### Requirement: Select default device
The system SHALL automatically select a default device for inference when multiple are available.

#### Scenario: Multiple GPUs present
- **WHEN** multiple GPU adapters are available
- **THEN** system selects the first high-performance adapter as default
