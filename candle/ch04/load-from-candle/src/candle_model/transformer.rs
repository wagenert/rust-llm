use candle_core::Tensor;
use candle_nn::{LayerNorm, Module, VarBuilder, layer_norm};

use crate::candle_model::{attention::Gpt2Attention, mlp::Gpt2Mlp};

pub struct Gpt2Block {
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    mlp: Gpt2Mlp,
    attention: Gpt2Attention,
}

impl Gpt2Block {
    pub fn new(n_head: usize, n_embed: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let layer_norm_1 = layer_norm(768, 1e-5, vb.pp("ln_1"))?;
        let layer_norm_2 = layer_norm(768, 1e-5, vb.pp("ln_2"))?;
        let attention = Gpt2Attention::new(n_head, n_embed, vb.pp("attn"))?;
        let mlp = Gpt2Mlp::new(n_embed, vb.pp("mlp"))?;
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
