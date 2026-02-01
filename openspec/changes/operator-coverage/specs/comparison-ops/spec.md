## ADDED Requirements

### Requirement: Element-wise comparison operations
The runtime SHALL support element-wise comparison operations that return boolean tensors.

Supported comparisons:
- `Equal`: a == b
- `Greater`: a > b
- `Less`: a < b
- `GreaterOrEqual`: a >= b
- `LessOrEqual`: a <= b

#### Scenario: Equal comparison
- **WHEN** executor encounters Equal node with inputs [1, 2, 3] and [1, 5, 3]
- **THEN** output tensor SHALL be [true, false, true]

#### Scenario: Greater comparison
- **WHEN** executor encounters Greater node with inputs [1, 5, 3] and [2, 2, 2]
- **THEN** output tensor SHALL be [false, true, true]

#### Scenario: Comparison with broadcasting
- **WHEN** executor encounters Less node with input shape [3, 4] and scalar 0.5
- **THEN** output tensor SHALL have shape [3, 4] with element-wise comparison against 0.5

### Requirement: Special value detection
The runtime SHALL support detecting special floating-point values.

- `IsInf`: Detect positive or negative infinity
- `IsNaN`: Detect Not-a-Number values

#### Scenario: IsInf detection
- **WHEN** executor encounters IsInf node with input [1.0, inf, -inf, 0.0]
- **THEN** output tensor SHALL be [false, true, true, false]

#### Scenario: IsNaN detection
- **WHEN** executor encounters IsNaN node with input [1.0, NaN, 0.0]
- **THEN** output tensor SHALL be [false, true, false]

### Requirement: Binary element-wise operations
The runtime SHALL support additional binary operations beyond Add/Sub/Mul/Div.

- `Pow`: Exponentiation (a^b)
- `Max`: Element-wise maximum
- `Min`: Element-wise minimum
- `Mod`: Modulo operation

#### Scenario: Pow operation
- **WHEN** executor encounters Pow node with inputs [2, 3, 4] and [2, 2, 2]
- **THEN** output tensor SHALL be [4, 9, 16]

#### Scenario: Element-wise Max
- **WHEN** executor encounters Max node with inputs [1, 5, 3] and [2, 2, 4]
- **THEN** output tensor SHALL be [2, 5, 4]

#### Scenario: Mod operation
- **WHEN** executor encounters Mod node with inputs [5, 7, 9] and [3, 3, 3]
- **THEN** output tensor SHALL be [2, 1, 0]
