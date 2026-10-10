use burn::prelude::*;
use burn::nn::LayerNormConfig;
use burn::{module::Module, nn::LayerNorm, tensor::backend::Backend};

use crate::model::attention::{CasualSelfAttention, CasualSelfAttentionConfig};
use crate::model::mlp::{Mlp, MlpConfig};
use crate::Gpt2ModelConfig;

#[derive(Debug)]
pub struct TransformerBlockConfig {
    d_model: usize,
    epsilon: f64,
    pub qkv_bias: bool,
    pub drop_rate: f64,
    pub n_embed: usize,
    pub n_head: usize,
    pub max_seq_len: usize,
}

impl TransformerBlockConfig {
    pub fn new(config: &Gpt2ModelConfig) -> Self {
        Self {
            d_model: config.n_embed,
            epsilon: config.layer_norm_epsilon,
            qkv_bias: config.qkv_bias,
            drop_rate: config.drop_rate,
            n_embed: config.n_embed,
            n_head: config.n_heads,
            max_seq_len: config.max_seq_len,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock::new(&self, device)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: CasualSelfAttention<B>,
    ln_2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &TransformerBlockConfig, device: &B::Device) -> Self {
        let ln_config = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.epsilon)
            .with_bias(config.qkv_bias);
        let ln_1 = ln_config.init(device);
        let ln_2 = ln_config.init(device);
        let attn = CasualSelfAttentionConfig::new(config).init(device);
        let mlp = MlpConfig::new(config).init(device);
        Self { ln_1, ln_2, attn, mlp }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let x = self.ln_1.forward(input);
        let x = self.attn.forward(x);
        let x = x + residual;
        let residual = x.clone();
        let x = self.ln_2.forward(x);
        let x = self.mlp.forward(x);
        x + residual
    }
}
