use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::backend::Backend,
};

use crate::transformer::TransformerBlockConfig;

#[derive(Debug)]
pub struct CasualSelfAttentionConfig {
    pub drop_rate: f64,
    pub n_embed: usize,
    pub max_seq_len: usize,
    pub n_head: usize,
}

impl CasualSelfAttentionConfig {
    pub fn new(config: &TransformerBlockConfig) -> Self {
        Self {
            drop_rate: config.drop_rate,
            n_embed: config.n_embed,
            n_head: config.n_head,
            max_seq_len: config.max_seq_len,
        }
    }

    pub fn head_dim(&self) -> usize {
        assert_eq!(
            self.n_embed % self.n_head,
            0,
            "n_embd ({}) must be divisible by n_head ({})",
            self.n_embed,
            self.n_head
        );
        self.n_embed / self.n_head
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> CasualSelfAttention<B> {
        CasualSelfAttention::new(self, device)
    }
}
#[derive(Module, Debug)]
pub struct CasualSelfAttention<B: Backend> {
    c_attn: Linear<B>,
    c_proj: Linear<B>,
    attn_drop: Dropout,
    resid_drop: Dropout,
    n_head: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl<B: Backend> CasualSelfAttention<B> {
    pub fn new(config: &CasualSelfAttentionConfig, device: &B::Device) -> Self {
        let attn_drop = DropoutConfig::new(config.drop_rate).init();
        let resid_drop = DropoutConfig::new(config.drop_rate).init();
        let c_attn = LinearConfig::new(config.n_embed, 3 * config.n_embed).init(device);
        let c_proj = LinearConfig::new(config.n_embed, config.n_embed).init(device);
        Self {
            attn_drop,
            resid_drop,
            c_attn,
            c_proj,
            n_head: config.n_head,
            head_dim: config.head_dim(),
            max_seq_len: config.max_seq_len,
        }
    }
}
