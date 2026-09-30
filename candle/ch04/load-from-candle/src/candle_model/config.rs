use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Gpt2Config {
    pub n_head: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    attn_pdrop: f32,
    n_ctx: usize,
    resid_pdrop: f32,
    layer_norm_epsilon: f32,
}