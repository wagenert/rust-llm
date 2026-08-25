use crate::GptModel;
use burn::{config::Config, tensor::backend::Backend};

pub struct GptConfig124M {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f64,
    pub qkv_bias: bool,
}

#[derive(Config, Debug)]
pub struct GptConfig {
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

impl GptConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GptModel<B> {
        GptModel::new(self, device)
    }
}
