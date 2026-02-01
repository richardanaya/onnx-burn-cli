## ADDED Requirements

### Requirement: LayerNorm operation
The runtime SHALL support layer normalization.

#### Scenario: LayerNorm forward pass
- **WHEN** executor encounters LayerNorm node with input shape [2, 3, 4], normalized_shape=[4], gamma and beta parameters
- **THEN** output tensor SHALL have shape [2, 3, 4] with normalization applied over last dimension

#### Scenario: LayerNorm with epsilon
- **WHEN** executor encounters LayerNorm node with epsilon=1e-5
- **THEN** normalization SHALL use epsilon for numerical stability in variance computation

### Requirement: InstanceNorm operation
The runtime SHALL support instance normalization.

#### Scenario: InstanceNorm forward pass
- **WHEN** executor encounters InstanceNorm node with input shape [2, 64, 32, 32]
- **THEN** output tensor SHALL have shape [2, 64, 32, 32] with normalization per instance per channel

### Requirement: GroupNorm operation
The runtime SHALL support group normalization.

#### Scenario: GroupNorm forward pass
- **WHEN** executor encounters GroupNorm node with input shape [2, 64, 32, 32], num_groups=8
- **THEN** output tensor SHALL have shape [2, 64, 32, 32] with normalization applied per group (8 groups of 8 channels each)

#### Scenario: GroupNorm divides channels evenly
- **WHEN** executor encounters GroupNorm node with 64 channels and num_groups=8
- **THEN** each group SHALL contain exactly 8 channels
