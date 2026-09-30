use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    tensor::backend::Backend,
};

use crate::transformer::TransformerBlockConfig;

#[derive(Debug)]
pub struct MlpConfig {
    drop_rate: f64,
    n_embed: usize,
    qkv_bias: bool,
}

impl MlpConfig {
    pub fn new(config: &TransformerBlockConfig) -> Self {
        Self {
            drop_rate: config.drop_rate,
            n_embed: config.n_embed,
            qkv_bias: config.qkv_bias,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp::new(self, device)
    }

    pub fn mlp_dim(&self) -> usize {
        4 * self.n_embed
    }
}
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    act: Gelu,
    c_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &MlpConfig, device: &B::Device) -> Self {
        let dropout = DropoutConfig::new(config.drop_rate).init();
        let act = Gelu::new();
        let c_fc = LinearConfig::new(config.n_embed, config.mlp_dim())
            .with_bias(config.qkv_bias)
            .init(device);
        let c_proj = LinearConfig::new(config.mlp_dim(), config.n_embed)
            .with_bias(config.qkv_bias)
            .init(device);
        Self {
            dropout,
            act,
            c_fc,
            c_proj,
        }
    }
}
