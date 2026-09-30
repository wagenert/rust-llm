use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_nn::attention::cpu_flash::flash_attn;

use crate::Gpt2Config;
use crate::candle_model::linear::Gpt2Linear;

pub struct Gpt2Attention {
    c_proj: Gpt2Linear,
    c_attn: Gpt2Linear,
    n_head: usize,
}

impl Gpt2Attention {
    pub fn new(cfg: &Gpt2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let c_proj = Gpt2Linear::new(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        let c_attn = Gpt2Linear::new(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?;
        let model = Self {
            c_proj,
            c_attn,
            n_head: cfg.n_head,
        };
        Ok(model)
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, n_embed) = x.dims3()?;
        let head_dim = n_embed / self.n_head;

        let qkv = self.c_attn.forward(&x)?;
        let chunks = qkv.chunk(3, candle_core::D::Minus1)?;

        let reshape_heads = |t: &Tensor| t.reshape((batch, seq_len, self.n_head, head_dim));
        let q = reshape_heads(&chunks[0])?;
        let k = reshape_heads(&chunks[1])?;
        let v = reshape_heads(&chunks[2])?;
        let attn_mask = candle_nn::attention::AttnMask::Causal { kv_offset: 0 };
        let softmax_scale = 1. / (head_dim as f32).sqrt();
        let attention = flash_attn::<f32>(&q, &k, &v, softmax_scale, attn_mask, None, None)?;
        self.c_proj.forward(&attention.reshape((batch, seq_len, n_embed))?)
    }
}