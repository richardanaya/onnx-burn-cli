## ADDED Requirements

### Requirement: Dimension manipulation operations
The runtime SHALL support operations that add or remove tensor dimensions.

- `Squeeze`: Remove dimensions of size 1
- `Unsqueeze`: Insert dimension of size 1 at specified axis

#### Scenario: Squeeze removes size-1 dimensions
- **WHEN** executor encounters Squeeze node with input shape [1, 3, 1, 4] and axes=[0, 2]
- **THEN** output tensor SHALL have shape [3, 4]

#### Scenario: Unsqueeze adds dimension
- **WHEN** executor encounters Unsqueeze node with input shape [3, 4] and axis=0
- **THEN** output tensor SHALL have shape [1, 3, 4]

#### Scenario: Unsqueeze at negative axis
- **WHEN** executor encounters Unsqueeze node with input shape [3, 4] and axis=-1
- **THEN** output tensor SHALL have shape [3, 4, 1]

### Requirement: Transpose operation
The runtime SHALL support tensor transposition with arbitrary permutation.

#### Scenario: Transpose with explicit permutation
- **WHEN** executor encounters Transpose node with input shape [2, 3, 4] and perm=[2, 0, 1]
- **THEN** output tensor SHALL have shape [4, 2, 3]

#### Scenario: Default transpose reverses dimensions
- **WHEN** executor encounters Transpose node with input shape [2, 3, 4] and no perm specified
- **THEN** output tensor SHALL have shape [4, 3, 2]

### Requirement: Concatenation operation
The runtime SHALL support concatenating multiple tensors along a specified axis.

#### Scenario: Concat along axis 0
- **WHEN** executor encounters Concat node with inputs of shapes [2, 3] and [4, 3], axis=0
- **THEN** output tensor SHALL have shape [6, 3]

#### Scenario: Concat along axis 1
- **WHEN** executor encounters Concat node with inputs of shapes [2, 3] and [2, 5], axis=1
- **THEN** output tensor SHALL have shape [2, 8]

#### Scenario: Concat multiple tensors
- **WHEN** executor encounters Concat node with 3 inputs of shape [2, 3] each, axis=0
- **THEN** output tensor SHALL have shape [6, 3]

### Requirement: Split operation
The runtime SHALL support splitting a tensor into multiple parts along an axis.

#### Scenario: Split into equal parts
- **WHEN** executor encounters Split node with input shape [6, 4], axis=0, num_outputs=3
- **THEN** outputs SHALL be 3 tensors each with shape [2, 4]

#### Scenario: Split with explicit sizes
- **WHEN** executor encounters Split node with input shape [10, 4], axis=0, split=[2, 3, 5]
- **THEN** outputs SHALL be tensors with shapes [2, 4], [3, 4], [5, 4]

### Requirement: Slice operation
The runtime SHALL support extracting a sub-tensor from specified ranges.

#### Scenario: Slice with start and end
- **WHEN** executor encounters Slice node with input shape [10, 20], starts=[2, 5], ends=[8, 15], axes=[0, 1]
- **THEN** output tensor SHALL have shape [6, 10] (elements from [2:8, 5:15])

#### Scenario: Slice with step
- **WHEN** executor encounters Slice node with input shape [10], starts=[0], ends=[10], steps=[2]
- **THEN** output tensor SHALL contain every 2nd element, shape [5]

### Requirement: Expand operation
The runtime SHALL support broadcasting a tensor to a larger shape.

#### Scenario: Expand to broadcast
- **WHEN** executor encounters Expand node with input shape [1, 3] and target shape [4, 3]
- **THEN** output tensor SHALL have shape [4, 3] with input broadcast across dimension 0

### Requirement: Tile operation
The runtime SHALL support repeating a tensor along each dimension.

#### Scenario: Tile tensor
- **WHEN** executor encounters Tile node with input shape [2, 3] and repeats=[2, 3]
- **THEN** output tensor SHALL have shape [4, 9]

### Requirement: Pad operation
The runtime SHALL support padding tensors with constant values.

#### Scenario: Pad with zeros
- **WHEN** executor encounters Pad node with input shape [3, 3], pads=[1, 1, 1, 1] (top, bottom, left, right), mode=constant, value=0
- **THEN** output tensor SHALL have shape [5, 5] with zeros around the edges

### Requirement: Shape and Size operations
The runtime SHALL support querying tensor shape information.

#### Scenario: Shape returns dimension sizes
- **WHEN** executor encounters Shape node with input of shape [2, 3, 4]
- **THEN** output SHALL be 1D tensor containing [2, 3, 4]

#### Scenario: Size returns total elements
- **WHEN** executor encounters Size node with input of shape [2, 3, 4]
- **THEN** output SHALL be scalar tensor containing 24
