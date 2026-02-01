## ADDED Requirements

### Requirement: Atan operator computes element-wise arctangent

The system SHALL implement the Atan operator that computes the element-wise arctangent (inverse tangent) of the input tensor. Output values SHALL be in the range [-pi/2, pi/2].

#### Scenario: Atan of positive values
- **WHEN** Atan node receives input tensor [0.0, 1.0, 1000.0]
- **THEN** the system SHALL output approximately [0.0, 0.785, 1.570] (0, pi/4, ~pi/2)

#### Scenario: Atan of negative values
- **WHEN** Atan node receives input tensor [-1.0, -1000.0]
- **THEN** the system SHALL output approximately [-0.785, -1.570] (-pi/4, ~-pi/2)

#### Scenario: Atan preserves tensor shape
- **WHEN** Atan node receives input tensor of shape [2, 3, 4]
- **THEN** the system SHALL output tensor of same shape [2, 3, 4]
