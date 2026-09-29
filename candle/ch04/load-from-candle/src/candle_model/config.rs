use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Gpt2Config {
    n_head: usize,
    n_embd: usize,
    vocab_size: usize,
    attn_pdrop: f32,
    n_ctx: usize,
    resid_pdrop: f32,
    layer_norm_epsilon: f32,
}