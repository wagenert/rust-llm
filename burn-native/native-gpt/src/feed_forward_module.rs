use burn::module::Module;
use burn::nn::{Gelu, Linear};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[derive(Debug, Module)]
pub struct FeedForward<B: AutodiffBackend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Gelu,
}
