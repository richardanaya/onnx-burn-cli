## ADDED Requirements

### Requirement: Range operator generates sequence tensor

The system SHALL implement the Range operator that generates a 1D tensor containing a sequence of numbers starting from `start`, incrementing by `delta`, up to but not including `limit`.

#### Scenario: Integer range generation
- **WHEN** Range node receives start=0, limit=5, delta=1
- **THEN** the system SHALL output tensor [0, 1, 2, 3, 4]

#### Scenario: Float range generation
- **WHEN** Range node receives start=0.0, limit=1.0, delta=0.25
- **THEN** the system SHALL output tensor [0.0, 0.25, 0.5, 0.75]

#### Scenario: Negative delta range
- **WHEN** Range node receives start=10, limit=0, delta=-2
- **THEN** the system SHALL output tensor [10, 8, 6, 4, 2]

#### Scenario: Empty range
- **WHEN** Range node receives start=5, limit=5, delta=1
- **THEN** the system SHALL output empty tensor with shape [0]
