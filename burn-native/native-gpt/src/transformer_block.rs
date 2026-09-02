use crate::feed_forward_module::FeedForward;
use burn::module::Module;
use burn::nn::{LayerNorm, modules::attention::MultiHeadAttention};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[derive(Debug, Module)]
pub struct TransformerBlock<B: AutodiffBackend> {
    layer_norm_in: LayerNorm<B>,
    layer_norm_out: LayerNorm<B>,
    multi_head_attention: MultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
}
