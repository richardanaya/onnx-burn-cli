## ADDED Requirements

### Requirement: Average pooling operations
The runtime SHALL support average pooling in 1D and 2D.

- `AvgPool1d`: 1D average pooling
- `AvgPool2d`: 2D average pooling

#### Scenario: AvgPool2d basic
- **WHEN** executor encounters AvgPool2d node with input shape [1, 1, 4, 4], kernel_size=[2, 2], stride=[2, 2]
- **THEN** output tensor SHALL have shape [1, 1, 2, 2] with averaged values

#### Scenario: AvgPool2d with padding
- **WHEN** executor encounters AvgPool2d node with input shape [1, 3, 224, 224], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]
- **THEN** output tensor SHALL have shape [1, 3, 224, 224] (same size with padding)

#### Scenario: AvgPool1d
- **WHEN** executor encounters AvgPool1d node with input shape [1, 64, 100], kernel_size=3, stride=2
- **THEN** output tensor SHALL have shape [1, 64, 49]

### Requirement: MaxPool1d operation
The runtime SHALL support 1D max pooling.

#### Scenario: MaxPool1d basic
- **WHEN** executor encounters MaxPool1d node with input shape [1, 64, 100], kernel_size=3, stride=2
- **THEN** output tensor SHALL have shape [1, 64, 49] with max values from each window
