use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::candle_model::linear::Gpt2Linear;

pub struct Gpt2Attention {
    c_proj: Gpt2Linear,
    c_attn: Gpt2Linear,
    n_head: usize,
}

impl Gpt2Attention {
    pub fn new(n_head: usize, n_embed: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let c_proj = Gpt2Linear::new(n_embed, n_embed, vb.pp("c_proj"))?;
        let c_attn = Gpt2Linear::new(n_embed, 3 * n_embed, vb.pp("c_attn"))?;
        let model = Self {
            c_proj,
            c_attn,
            n_head,
        };
        Ok(model)
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let qkv = self.c_attn.forward(&x)?;
        let chunks = qkv.chunk(3, candle_core::D::Minus1)?;
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);
        let attn_mask = candle_nn::attention::AttnMask::Causal { kv_offset: 0 };
        let softmax_scale = (*k.dims().last().unwrap() as f32).sqrt();
        let attention = candle_nn::attention::flash_attn::<f32>(q, k, v, softmax_scale, attn_mask, None, None)?;
        self.c_proj.forward(&attention)
    }
}