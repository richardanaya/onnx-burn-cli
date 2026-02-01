## ADDED Requirements

### Requirement: Reduction operations
The runtime SHALL support reducing tensor values along specified axes.

Supported reductions:
- `ReduceSum`: Sum of elements
- `ReduceMean`: Mean of elements
- `ReduceMax`: Maximum element
- `ReduceMin`: Minimum element
- `ReduceProd`: Product of elements

#### Scenario: ReduceSum along axis
- **WHEN** executor encounters ReduceSum node with input [[1, 2], [3, 4]] and axis=1
- **THEN** output tensor SHALL be [3, 7] (sum along rows)

#### Scenario: ReduceMean along axis
- **WHEN** executor encounters ReduceMean node with input [[1, 2, 3], [4, 5, 6]] and axis=1
- **THEN** output tensor SHALL be [2.0, 5.0] (mean along rows)

#### Scenario: ReduceMax along axis with keepdims
- **WHEN** executor encounters ReduceMax node with input shape [2, 3, 4], axis=1, keepdims=true
- **THEN** output tensor SHALL have shape [2, 1, 4]

#### Scenario: ReduceMax along axis without keepdims
- **WHEN** executor encounters ReduceMax node with input shape [2, 3, 4], axis=1, keepdims=false
- **THEN** output tensor SHALL have shape [2, 4]

#### Scenario: Reduce over all axes
- **WHEN** executor encounters ReduceSum node with input [[1, 2], [3, 4]] and no axes specified
- **THEN** output tensor SHALL be scalar [10]

### Requirement: Argmax and Argmin operations
The runtime SHALL support finding indices of maximum and minimum values.

#### Scenario: Argmax along axis
- **WHEN** executor encounters Argmax node with input [[1, 5, 3], [7, 2, 4]] and axis=1
- **THEN** output tensor SHALL be [1, 0] (indices of max in each row)

#### Scenario: Argmin along axis
- **WHEN** executor encounters Argmin node with input [[1, 5, 3], [7, 2, 4]] and axis=1
- **THEN** output tensor SHALL be [0, 1] (indices of min in each row)

#### Scenario: Argmax with keepdims
- **WHEN** executor encounters Argmax node with input shape [2, 3], axis=1, keepdims=true
- **THEN** output tensor SHALL have shape [2, 1]
