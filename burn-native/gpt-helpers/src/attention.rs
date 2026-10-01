use burn::{nn::{Dropout, DropoutConfig, Linear, LinearConfig}, prelude::*, tensor::activation::softmax};


use crate::transformer::TransformerBlockConfig;

#[derive(Debug)]
pub struct CasualSelfAttentionConfig {
    pub drop_rate: f64,
    pub n_embed: usize,
    pub max_seq_len: usize,
    pub n_head: usize,
    pub qkv_bias: bool,
}

impl CasualSelfAttentionConfig {
    pub fn new(config: &TransformerBlockConfig) -> Self {
        Self {
            drop_rate: config.drop_rate,
            n_embed: config.n_embed,
            n_head: config.n_head,
            max_seq_len: config.max_seq_len,
            qkv_bias: config.qkv_bias,
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
        let c_attn = LinearConfig::new(config.n_embed, 3 * config.n_embed)
            .with_bias(config.qkv_bias)
            .init(device);
        let c_proj = LinearConfig::new(config.n_embed, config.n_embed)
            .with_bias(config.qkv_bias)
            .init(device);
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

    pub fn forward(&self, input: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        let [batch_size, seq_length, _] = input.dims();
        // 1. Combined QKV projection
        // x: [B, T, n_embd]  →  qkv: [B, T, 3*n_embd]
        let qkv = self.c_attn.forward(input);
        let n_embd = self.n_head * self.head_dim;
        // Split along last dimension into Q, K, V each [B, T, n_embd]
        let q = qkv.clone().slice([0..batch_size, 0..seq_length, 0..n_embd]);
        let k = qkv.clone().slice([0..batch_size, 0..seq_length, n_embd..2 * n_embd]);
        let v = qkv.slice([0..batch_size, 0..seq_length, 2 * n_embd..3 * n_embd]);
        
        // 2. Reshape to multi-head form
        // [B, T, n_embd] → [B, T, n_head, head_dim] → [B, n_head, T, head_dim]
        let q = q
            .reshape([batch_size, seq_length, self.n_head, self.head_dim])
            .swap_dims(1, 2); // [B, n_head, T, head_dim]
        let k = k
            .reshape([batch_size, seq_length, self.n_head, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_length, self.n_head, self.head_dim])
            .swap_dims(1, 2);

        // 3. Scaled dot-product attention + causal mask
        // scores: [B, n_head, T, T]
        let scale = (self.head_dim as f64).sqrt();
        // k^T: [B, n_head, head_dim, T]
        let k_t = k.transpose();
        let scores = q.matmul(k_t) / scale;

        // Causal mask: upper triangle is -inf so future tokens are invisible.
        // mask[i,j] = -inf if j > i
        let mask = self.causal_mask(seq_length, &scores.device());
        let scores = scores + mask;

        // Softmax over the last (key) dimension
        let attn_weights = softmax(scores, 3); // [B, n_head, T, T]
        let attn_weights = self.attn_drop.forward(attn_weights);

        // Weighted sum of values
        // [B, n_head, T, T] × [B, n_head, T, head_dim] → [B, n_head, T, head_dim]
        let context = attn_weights.matmul(v);

        // 4. Re-assemble heads
        // [B, n_head, T, head_dim] → [B, T, n_head, head_dim] → [B, T, n_embd]
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length, self.n_head * self.head_dim]);

        // 5. Output projection
        let out = self.c_proj.forward(context);
        self.resid_drop.forward(out)
    }

    fn causal_mask(&self, seq_length: usize, device: &B::Device) -> Tensor<B, 4, Float> {
        let mut mask = Tensor::<B, 2, Float>::zeros([seq_length, seq_length], device);
        let boolean_mask = Tensor::<B, 2, Bool>::triu_mask([seq_length, seq_length], 1, device);
        mask = mask.mask_fill(boolean_mask, f64::NEG_INFINITY);
        mask.unsqueeze::<4>()
    }
}
