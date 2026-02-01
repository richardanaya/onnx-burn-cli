use crate::runtime::tensor::DynTensor;
use burn::prelude::Backend;
use std::collections::HashMap;

/// Storage for tensor values by name, keyed by tensor output names
pub struct ValueStore<B: Backend> {
    values: HashMap<String, DynTensor<B>>,
}

impl<B: Backend> Default for ValueStore<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ValueStore<B> {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<&DynTensor<B>> {
        self.values.get(name)
    }

    /// Insert a tensor with a given name
    pub fn insert(&mut self, name: String, tensor: DynTensor<B>) {
        self.values.insert(name, tensor);
    }

    /// Remove a tensor
    pub fn remove(&mut self, name: &str) -> Option<DynTensor<B>> {
        self.values.remove(name)
    }

    /// Get the names of all stored tensors
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.values.keys()
    }

    /// Check if a tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Extend with multiple tensors
    pub fn extend<I: IntoIterator<Item = (String, DynTensor<B>)>>(&mut self, iter: I) {
        self.values.extend(iter);
    }
}
