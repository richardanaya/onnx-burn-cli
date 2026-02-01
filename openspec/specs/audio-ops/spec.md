## ADDED Requirements

### Requirement: STFT operator computes Short-Time Fourier Transform

The system SHALL implement the STFT (Short-Time Fourier Transform) operator that computes the frequency-domain representation of a signal using overlapping windowed segments.

The output SHALL have shape [batch, num_frames, fft_size/2+1, 2] where the last dimension contains [real, imaginary] components.

#### Scenario: Basic STFT computation
- **WHEN** STFT node receives signal tensor of shape [batch, signal_length], frame_length, frame_step, and optional window tensor
- **THEN** the system SHALL compute windowed DFT for each frame and output complex frequency coefficients

#### Scenario: STFT with Hann window
- **WHEN** STFT node receives window tensor containing Hann window coefficients
- **THEN** the system SHALL apply the window to each frame before computing DFT

#### Scenario: STFT frame calculation
- **WHEN** signal has length L, frame_length is N, and frame_step is H
- **THEN** the system SHALL compute num_frames = floor((L - N) / H) + 1 frames

#### Scenario: STFT output format
- **WHEN** onesided attribute is 1 (default)
- **THEN** the system SHALL output only positive frequencies (fft_size/2 + 1 bins)

### Requirement: STFT handles batched inputs

The system SHALL support batched signal inputs where the first dimension is batch size.

#### Scenario: Batched STFT
- **WHEN** STFT receives signal of shape [B, L] where B > 1
- **THEN** the system SHALL compute STFT independently for each batch element and output shape [B, num_frames, num_bins, 2]
