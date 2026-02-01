## ADDED Requirements

### Requirement: ScatterND operator scatters updates into tensor

The system SHALL implement the ScatterND operator that scatters values from `updates` tensor into a copy of `data` tensor at indices specified by `indices` tensor.

For each entry in indices, the corresponding update slice replaces the data at that index position.

#### Scenario: Basic ScatterND update
- **WHEN** ScatterND node receives data tensor of shape [4, 4, 4], indices tensor of shape [2, 1] with values [[0], [2]], and updates tensor of shape [2, 4, 4]
- **THEN** the system SHALL copy data and replace slices at indices 0 and 2 with the update slices

#### Scenario: ScatterND with multi-dimensional indices
- **WHEN** ScatterND node receives data of shape [4, 5, 6], indices of shape [3, 2] specifying 2D coordinates, and updates of shape [3, 6]
- **THEN** the system SHALL scatter each update row to the specified [i, j, :] location

#### Scenario: ScatterND preserves unchanged elements
- **WHEN** ScatterND updates only a subset of data tensor locations
- **THEN** the system SHALL preserve all other data elements unchanged in the output

### Requirement: ScatterND supports reduction modes

The system SHALL support the optional `reduction` attribute with values "none" (default), "add", "mul", "max", "min".

#### Scenario: ScatterND with add reduction
- **WHEN** ScatterND node has reduction="add" and multiple indices point to same location
- **THEN** the system SHALL add updates to existing values instead of replacing
