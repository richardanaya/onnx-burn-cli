## ADDED Requirements

### Requirement: Unary math operations
The runtime SHALL support single-input mathematical operations that apply element-wise to tensors.

Supported operations:
- `Abs`: Absolute value
- `Neg`: Negation
- `Sqrt`: Square root
- `Exp`: Exponential (e^x)
- `Log`: Natural logarithm
- `Ceil`: Ceiling (round up)
- `Floor`: Floor (round down)
- `Round`: Round to nearest integer
- `Sign`: Sign function (-1, 0, or 1)
- `Reciprocal`: 1/x
- `Sin`, `Cos`, `Tan`: Trigonometric functions
- `Sinh`, `Cosh`: Hyperbolic functions
- `Erf`: Error function

#### Scenario: Apply abs to tensor
- **WHEN** executor encounters Abs node with input tensor containing [-1.0, 2.0, -3.0]
- **THEN** output tensor SHALL contain [1.0, 2.0, 3.0]

#### Scenario: Apply sqrt to tensor
- **WHEN** executor encounters Sqrt node with input tensor containing [4.0, 9.0, 16.0]
- **THEN** output tensor SHALL contain [2.0, 3.0, 4.0]

#### Scenario: Apply exp to tensor
- **WHEN** executor encounters Exp node with input tensor containing [0.0, 1.0]
- **THEN** output tensor SHALL contain [1.0, 2.718...] (e^0 and e^1)

#### Scenario: Preserve tensor shape
- **WHEN** executor applies any unary math operation to tensor of shape [2, 3, 4]
- **THEN** output tensor SHALL have shape [2, 3, 4]

### Requirement: Additional activation functions
The runtime SHALL support activation functions beyond Relu/Sigmoid/Softmax.

Supported activations:
- `Tanh`: Hyperbolic tangent
- `Gelu`: Gaussian Error Linear Unit
- `LeakyRelu`: Leaky ReLU with configurable alpha
- `HardSigmoid`: Piecewise linear approximation of sigmoid
- `HardSwish`: Hard swish activation
- `Prelu`: Parametric ReLU with learned alpha
- `LogSoftmax`: Log of softmax

#### Scenario: Apply tanh activation
- **WHEN** executor encounters Tanh node with input tensor
- **THEN** output tensor SHALL contain tanh(x) for each element, values in range [-1, 1]

#### Scenario: Apply gelu activation
- **WHEN** executor encounters Gelu node with input tensor
- **THEN** output tensor SHALL contain GELU(x) = x * Φ(x) where Φ is the CDF of standard normal

#### Scenario: Apply leaky_relu with alpha
- **WHEN** executor encounters LeakyRelu node with alpha=0.01 and input [-1.0, 0.0, 1.0]
- **THEN** output tensor SHALL contain [-0.01, 0.0, 1.0]

### Requirement: Trivial pass-through operations
The runtime SHALL support operations that require minimal or no computation.

- `Identity`: Return input unchanged
- `Dropout`: No-op in inference mode (pass-through)
- `Clip`: Clamp values to [min, max] range

#### Scenario: Identity pass-through
- **WHEN** executor encounters Identity node
- **THEN** output tensor SHALL be identical to input tensor

#### Scenario: Dropout in inference mode
- **WHEN** executor encounters Dropout node during inference
- **THEN** output tensor SHALL be identical to input tensor (no dropout applied)

#### Scenario: Clip values to range
- **WHEN** executor encounters Clip node with min=0, max=6 and input [-1.0, 3.0, 10.0]
- **THEN** output tensor SHALL contain [0.0, 3.0, 6.0]
