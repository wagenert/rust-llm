use candle_core::Tensor;
use candle_nn::{LayerNorm, Module, VarBuilder, layer_norm};

use crate::{Gpt2Config, candle_model::{attention::Gpt2Attention, mlp::Gpt2Mlp}};

pub struct Gpt2Block {
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    mlp: Gpt2Mlp,
    attention: Gpt2Attention,
}

impl Gpt2Block {
    pub fn new(cfg: &Gpt2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let layer_norm_1 = layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let layer_norm_2 = layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let attention = Gpt2Attention::new(cfg, vb.pp("attn"))?;
        let mlp = Gpt2Mlp::new(cfg.n_embd, vb.pp("mlp"))?;
        let model = Self {
            ln_1: layer_norm_1,
            ln_2: layer_norm_2,
            mlp,
            attention,
        };
        Ok(model)
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.ln_1.forward(&x)?;
        let x = self.attention.forward(&x)?;
        let x = x.broadcast_add(residual)?;
        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.broadcast_add(residual)
    }
}
