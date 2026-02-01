## ADDED Requirements

### Requirement: Load image files as input
The system SHALL load image files and convert them to input tensors.

#### Scenario: Load PNG image
- **WHEN** a PNG image path is provided with `-i` flag
- **THEN** system loads the image and converts it to a tensor

#### Scenario: Load JPEG image
- **WHEN** a JPEG image path is provided with `-i` flag
- **THEN** system loads the image and converts it to a tensor

#### Scenario: Image file not found
- **WHEN** an invalid image path is provided
- **THEN** system returns an error indicating the file was not found

### Requirement: Resize images to model input size
The system SHALL resize input images to match the model's expected input dimensions.

#### Scenario: Resize to model input shape
- **WHEN** model expects 224x224 input and a 512x512 image is provided
- **THEN** system resizes the image to 224x224

### Requirement: Normalize image pixel values
The system SHALL normalize image pixel values to the expected range.

#### Scenario: Standard ImageNet normalization
- **WHEN** image is loaded for a typical vision model
- **THEN** system normalizes pixels to [0,1] range and applies channel-wise mean/std normalization

### Requirement: Convert image to NCHW tensor format
The system SHALL convert images to NCHW (batch, channels, height, width) tensor format.

#### Scenario: HWC to NCHW conversion
- **WHEN** an RGB image (HWC format) is loaded
- **THEN** system converts to NCHW format with batch dimension of 1

### Requirement: Parse labels file
The system SHALL parse a labels file containing one class name per line.

#### Scenario: Load labels
- **WHEN** a labels file is provided with `--labels` flag
- **THEN** system reads the file and maps indices to class names

#### Scenario: Labels file not found
- **WHEN** an invalid labels file path is provided
- **THEN** system returns an error indicating the file was not found
