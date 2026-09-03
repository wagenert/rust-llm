// use crate::transformer_block::TransformerBlock;
use burn::module::Module;
use burn::nn::modules::transformer::{TransformerEncoder, TransformerEncoderConfig};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    PositionalEncoding, PositionalEncodingConfig,
};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[derive(Config, Debug)]
pub struct BurnModelConfig {
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 1024)]
    pub context_length: usize,
    #[config(default = 768)]
    pub emb_dim: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = 12)]
    pub n_layers: usize,
    #[config(default = 0.1)]
    pub drop_rate: f64,
    #[config(default = false)]
    pub qkv_bias: bool,
    pub transformer_encoder_config: TransformerEncoderConfig,
    pub positional_encoding_config: PositionalEncodingConfig,
    pub embedding_config: EmbeddingConfig,
}

impl BurnModelConfig {
    pub fn init<B: AutodiffBackend>(self, &device: &B::Device) -> BurnModel<B> {}
}

#[derive(Debug, Module)]
pub struct BurnModel<B: AutodiffBackend> {
    token_embedding: Embedding<B>,
    positional_embedding: PositionalEncoding<B>,
    dropout: Dropout,
    transformers: TransformerEncoder<B>,
    final_norm: LayerNorm<B>,
    output_layer: Linear<B>,
}

impl<B: AutodiffBackend> BurnModel<B> {
    pub fn new(config: BurnModelConfig, device: &B::Device) -> Self {
        let token_embedding = config.embedding_config.init(device);
        let positional_embedding = config.positional_encoding_config.init(device);
        let dropout = DropoutConfig::new(config.drop_rate).init();
        let transformers = config.transformer_encoder_config.init(device);
        let final_norm = LayerNormConfig::new(config.emb_dim).init(device);
        let output_layer = LinearConfig::new(config.emb_dim, config.vocab_size).init(device);

        Self {
            token_embedding,
            positional_embedding,
            dropout,
            transformers,
            final_norm,
            output_layer,
        }
    }
}
