## ADDED Requirements

### Requirement: And operator computes element-wise logical AND

The system SHALL implement the And operator that performs element-wise logical AND on two boolean tensors with NumPy-style broadcasting.

#### Scenario: Element-wise AND of same-shape tensors
- **WHEN** And node receives tensor A=[true, false, true] and tensor B=[true, true, false]
- **THEN** the system SHALL output tensor [true, false, false]

#### Scenario: AND with broadcasting
- **WHEN** And node receives tensor A of shape [2, 3] and tensor B of shape [3]
- **THEN** the system SHALL broadcast B and compute element-wise AND, outputting shape [2, 3]

#### Scenario: AND with scalar
- **WHEN** And node receives tensor A of shape [3] and scalar B=true
- **THEN** the system SHALL output tensor equal to A
