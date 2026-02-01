## ADDED Requirements

### Requirement: Gather operation
The runtime SHALL support gathering elements from a tensor using indices.

#### Scenario: Gather along axis
- **WHEN** executor encounters Gather node with input [[1, 2], [3, 4]], indices [0, 1, 0], axis=0
- **THEN** output tensor SHALL be [[1, 2], [3, 4], [1, 2]]

#### Scenario: Gather for embedding lookup
- **WHEN** executor encounters Gather node with embedding table shape [1000, 512] and indices [5, 10, 15]
- **THEN** output tensor SHALL have shape [3, 512] with embeddings at indices 5, 10, 15

### Requirement: GatherElements operation
The runtime SHALL support gathering elements using index tensors of same shape.

#### Scenario: GatherElements basic
- **WHEN** executor encounters GatherElements node with input [[1, 2], [3, 4]], indices [[0, 0], [1, 0]], axis=1
- **THEN** output tensor SHALL be [[1, 1], [4, 3]]

### Requirement: TopK operation
The runtime SHALL support finding top-k values and indices.

#### Scenario: TopK values and indices
- **WHEN** executor encounters TopK node with input [1, 5, 3, 9, 2] and k=3
- **THEN** output values SHALL be [9, 5, 3] and indices SHALL be [3, 1, 2]

#### Scenario: TopK along axis
- **WHEN** executor encounters TopK node with input shape [2, 5], k=2, axis=1
- **THEN** output values SHALL have shape [2, 2] with top-2 per row

### Requirement: NonZero operation
The runtime SHALL support finding indices of non-zero elements.

#### Scenario: NonZero indices
- **WHEN** executor encounters NonZero node with input [0, 1, 0, 2, 0]
- **THEN** output tensor SHALL contain indices [1, 3]

### Requirement: Where operation
The runtime SHALL support conditional element selection.

#### Scenario: Where conditional selection
- **WHEN** executor encounters Where node with condition [true, false, true], x=[1, 2, 3], y=[4, 5, 6]
- **THEN** output tensor SHALL be [1, 5, 3]

### Requirement: OneHot operation
The runtime SHALL support one-hot encoding.

#### Scenario: OneHot encoding
- **WHEN** executor encounters OneHot node with indices [0, 2, 1], depth=3
- **THEN** output tensor SHALL be [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

### Requirement: CumSum operation
The runtime SHALL support cumulative sum along an axis.

#### Scenario: CumSum along axis
- **WHEN** executor encounters CumSum node with input [1, 2, 3, 4] and axis=0
- **THEN** output tensor SHALL be [1, 3, 6, 10]

### Requirement: MatMul operation
The runtime SHALL support matrix multiplication with batching.

#### Scenario: MatMul 2D
- **WHEN** executor encounters MatMul node with inputs shape [3, 4] and [4, 5]
- **THEN** output tensor SHALL have shape [3, 5]

#### Scenario: MatMul batched
- **WHEN** executor encounters MatMul node with inputs shape [2, 3, 4] and [2, 4, 5]
- **THEN** output tensor SHALL have shape [2, 3, 5] (batch matmul)

### Requirement: Gemm operation
The runtime SHALL support general matrix multiplication with alpha, beta, and transpose options.

#### Scenario: Gemm basic (Y = alpha * A * B + beta * C)
- **WHEN** executor encounters Gemm node with A shape [3, 4], B shape [4, 5], C shape [5], alpha=1.0, beta=1.0
- **THEN** output tensor SHALL have shape [3, 5]

#### Scenario: Gemm with transA
- **WHEN** executor encounters Gemm node with transA=1, A shape [4, 3], B shape [4, 5]
- **THEN** A SHALL be transposed to [3, 4] before multiplication

### Requirement: Resize operation
The runtime SHALL support resizing tensors with interpolation.

#### Scenario: Resize with nearest neighbor
- **WHEN** executor encounters Resize node with input shape [1, 3, 224, 224], scales=[1, 1, 2, 2], mode=nearest
- **THEN** output tensor SHALL have shape [1, 3, 448, 448]

#### Scenario: Resize with bilinear interpolation
- **WHEN** executor encounters Resize node with input shape [1, 3, 224, 224], output_size=[112, 112], mode=linear
- **THEN** output tensor SHALL have shape [1, 3, 112, 112] with bilinear interpolation

### Requirement: Cast operation
The runtime SHALL support converting tensors between data types.

#### Scenario: Cast float to int
- **WHEN** executor encounters Cast node with input float tensor [1.5, 2.7, 3.2] and target type int64
- **THEN** output tensor SHALL be int64 tensor [1, 2, 3]

#### Scenario: Cast int to float
- **WHEN** executor encounters Cast node with input int tensor [1, 2, 3] and target type float32
- **THEN** output tensor SHALL be float32 tensor [1.0, 2.0, 3.0]
