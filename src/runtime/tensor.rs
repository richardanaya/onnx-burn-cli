use burn::prelude::Backend;
use burn::tensor::Tensor;
use std::fmt;

/// Dynamically-typed tensor wrapper.
/// All tensors are stored as rank-4 internally (padding with 1s for lower ranks).
/// Tracks original rank for correct reshaping on output.
#[derive(Clone)]
pub struct DynTensor<B: Backend> {
    pub tensor: Tensor<B, 4>,
    pub original_shape: Vec<usize>,
}

impl<B: Backend> DynTensor<B> {
    /// Create from a rank-1 tensor
    pub fn from_rank1(t: Tensor<B, 1>) -> Self {
        let shape = t.dims().to_vec();
        let tensor = t.reshape([1, 1, 1, shape[0]]);
        Self {
            tensor,
            original_shape: shape,
        }
    }

    /// Create from a rank-2 tensor
    pub fn from_rank2(t: Tensor<B, 2>) -> Self {
        let shape = t.dims().to_vec();
        let tensor = t.reshape([1, 1, shape[0], shape[1]]);
        Self {
            tensor,
            original_shape: shape,
        }
    }

    /// Create from a rank-3 tensor
    pub fn from_rank3(t: Tensor<B, 3>) -> Self {
        let shape = t.dims().to_vec();
        let tensor = t.reshape([1, shape[0], shape[1], shape[2]]);
        Self {
            tensor,
            original_shape: shape,
        }
    }

    /// Create from a rank-4 tensor
    pub fn from_rank4(t: Tensor<B, 4>) -> Self {
        let shape = t.dims().to_vec();
        Self {
            tensor: t,
            original_shape: shape,
        }
    }

    /// Get as rank-2 tensor (for matmul operations, etc.)
    pub fn as_rank2(&self) -> Tensor<B, 2> {
        let s = &self.original_shape;
        match s.len() {
            1 => self.tensor.clone().reshape([1, s[0]]),
            2 => self.tensor.clone().reshape([s[0], s[1]]),
            _ => panic!("Cannot convert rank-{} tensor to rank-2", s.len()),
        }
    }

    /// Get as rank-3 tensor (for 1D conv/pooling operations)
    pub fn as_rank3(&self) -> Tensor<B, 3> {
        let s = &self.original_shape;
        match s.len() {
            1 => self.tensor.clone().reshape([1, 1, s[0]]),
            2 => self.tensor.clone().reshape([1, s[0], s[1]]),
            3 => self.tensor.clone().reshape([s[0], s[1], s[2]]),
            _ => panic!("Cannot convert rank-{} tensor to rank-3", s.len()),
        }
    }

    /// Get as rank-4 tensor
    pub fn as_rank4(&self) -> Tensor<B, 4> {
        self.tensor.clone()
    }

    /// Get the original shape
    pub fn shape(&self) -> &[usize] {
        &self.original_shape
    }

    /// Get the original rank
    pub fn rank(&self) -> usize {
        self.original_shape.len()
    }
}

impl<B: Backend> fmt::Debug for DynTensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynTensor(shape: {:?}, rank: {})",
            self.original_shape,
            self.original_shape.len()
        )
    }
}
