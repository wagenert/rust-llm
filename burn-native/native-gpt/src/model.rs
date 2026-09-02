use crate::transformer_block::TransformerBlock;
use burn::module::Module;
use burn::nn::{Dropout, Embedding, LayerNorm, Linear};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[derive(Debug, Module)]
pub struct BurnModel<B: AutodiffBackend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    dropout: Dropout,
    transformers: Vec<TransformerBlock<B>>,
    final_norm: LayerNorm<B>,
    output_layer: Linear<B>,
}
