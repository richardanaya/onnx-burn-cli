## ADDED Requirements

### Requirement: LSTM operator executes forward pass

The system SHALL implement the LSTM (Long Short-Term Memory) operator according to ONNX specification. The LSTM SHALL compute hidden and cell states using input gates (i), output gates (o), forget gates (f), and cell gates (c) with the formula:

- `i = sigmoid(Wi*X + Ri*H + Wbi + Rbi)`
- `f = sigmoid(Wf*X + Rf*H + Wbf + Rbf)`
- `c = tanh(Wc*X + Rc*H + Wbc + Rbc)`
- `C_new = f * C_prev + i * c`
- `o = sigmoid(Wo*X + Ro*H + Wbo + Rbo)`
- `H_new = o * tanh(C_new)`

#### Scenario: Basic LSTM forward pass
- **WHEN** LSTM node receives input tensor X of shape [seq_length, batch, input_size] and initial hidden state H of shape [num_directions, batch, hidden_size]
- **THEN** the system SHALL output Y of shape [seq_length, num_directions, batch, hidden_size] and Y_h of shape [num_directions, batch, hidden_size]

#### Scenario: LSTM with default initial states
- **WHEN** LSTM node receives input tensor but no initial hidden/cell states
- **THEN** the system SHALL initialize hidden and cell states to zeros

#### Scenario: Bidirectional LSTM
- **WHEN** LSTM node has direction attribute set to "bidirectional"
- **THEN** the system SHALL compute forward and backward passes and concatenate outputs along the num_directions dimension

### Requirement: LSTM handles variable sequence lengths

The system SHALL support the optional sequence_lens input for batched sequences of different lengths.

#### Scenario: Padded batch with sequence lengths
- **WHEN** LSTM receives sequence_lens tensor specifying actual lengths per batch element
- **THEN** the system SHALL only process valid timesteps for each sequence and pad outputs appropriately
