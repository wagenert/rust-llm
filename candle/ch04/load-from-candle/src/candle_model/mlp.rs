use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::candle_model::linear::Gpt2Linear;

pub struct Gpt2Mlp {
    c_fc: Gpt2Linear,
    c_proj: Gpt2Linear,
}

impl Gpt2Mlp {
    pub fn new(n_embed: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let c_fc = Gpt2Linear::new(n_embed, n_embed * 4, vb.pp("c_fc"))?;
        let c_proj = Gpt2Linear::new(n_embed * 4, n_embed, vb.pp("c_proj"))?;
        let model = Self {
            c_fc,
            c_proj,
        };
        Ok(model)
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.c_fc.forward(&x)?;
        let x = x.gelu()?;
        self.c_proj.forward(&x)
    }
}