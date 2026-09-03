// use crate::transformer_block::TransformerBlock;
use burn::module::Module;
use burn::nn::modules::transformer::{TransformerEncoder, TransformerEncoderConfig};
use burn::nn::transformer::TransformerEncoderInput;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
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
}

impl BurnModelConfig {
    pub fn init<B: AutodiffBackend>(self, device: &B::Device) -> BurnModel<B> {
        BurnModel::new(self, device)
    }
}

#[derive(Debug, Module)]
pub struct BurnModel<B: AutodiffBackend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    dropout: Dropout,
    transformers: TransformerEncoder<B>,
    final_norm: LayerNorm<B>,
    output_layer: Linear<B>,
}

impl<B: AutodiffBackend> BurnModel<B> {
    pub fn new(config: BurnModelConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.emb_dim).init(device);
        let positional_embedding = EmbeddingConfig::new(config.context_length, config.emb_dim).init(device);
        let dropout = DropoutConfig::new(config.drop_rate).init();
        let transformers =
            TransformerEncoderConfig::new(config.emb_dim, config.emb_dim, config.n_heads, config.n_layers).init(device);
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

    pub fn forward(self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let input_shape = input.shape();
        let _batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let tok_embeds = self.token_embedding.forward(input.clone());
        let pos_input =
            Tensor::<B, 1, Int>::from_data(Vec::from_iter(0..seq_length).as_slice(), &input.device()).unsqueeze();
        let pos_embeds = self.positional_embedding.forward(pos_input);
        let x = tok_embeds + pos_embeds;
        let x = self.dropout.forward(x);
        let x = TransformerEncoderInput::new(x);
        let x = self.transformers.forward(x);
        let x = self.final_norm.forward(x);
        let x = self.output_layer.forward(x);
        x
    }
}
