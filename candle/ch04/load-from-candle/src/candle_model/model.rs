use candle_core::{Tensor, Module};
use candle_nn::{Embedding, LayerNorm, VarBuilder, embedding, layer_norm};

use crate::{Gpt2Config, candle_model::transformer::Gpt2Block};

pub struct Gpt2Model {
    wte: Embedding,
    wpe: Embedding,
    ln_f: LayerNorm,
    h: Vec<Gpt2Block>,
}

impl Gpt2Model {
    pub fn new(cfg: &Gpt2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = embedding(cfg.n_ctx, cfg.n_embd, vb.pp("wpe"))?;
        let ln_f = layer_norm(cfg.n_embd, cfg.layer_norm_epsilon , vb.pp("ln_f"))?;
        
        let transformer_block_weights = vb.set_prefix("h");
        let mut h = Vec::with_capacity(cfg.n_head);
        for i in 0..cfg.n_head {
            let transformer_block = Gpt2Block::new(cfg.n_embd, transformer_block_weights.pp(format!("{i}")))?;
            h.push(transformer_block);
        }
        let model = Self {
            wte,
            wpe,
            ln_f,
            h,
        };
        Ok(model)
    }

    pub fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        
        // 1. Token embeddings
        let tok_emb = self.wte.forward(input_ids)?;
        
        // 2. Generate position indices [0, 1, ..., seq_len-1]
        let pos = Tensor::arange(0u32, seq_len as u32, input_ids.device())?;
        let pos_emb = self.wpe.forward(&pos)?;
        
        // 3. Sum token and position vectors
        let mut x = tok_emb.broadcast_add(&pos_emb)?;
        
        // 4. Pass through sequentially numbered transformer blocks
        for block in &self.h {
            x = block.forward(&x)?;
        }
        
        // 5. Final Layer Normalization
        self.ln_f.forward(&x)
    }
}