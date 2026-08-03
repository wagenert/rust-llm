use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn_helpers::casual_attention::CasualAttention;
use burn::tensor::{backend::Backend, Tensor};
use burn::prelude::Float;

pub struct MultiHeadAttentionWrapper<B: Backend> {
    num_heads: usize,
    context_len: usize,
    casual_attention: Vec<CasualAttention<B>>,
}

impl<B: Backend> MultiHeadAttentionWrapper<B> {
    pub fn new(d_in: usize, d_out: usize, context_len: usize, dropout: f64, num_heads: usize, qkv_bias: bool, device: B::Device) -> Self {
        let casual_attention = (0..num_heads).map(
            |_| CasualAttention::<B>::new(d_in, d_out, dropout, qkv_bias, &device)
        ).collect();
        Self { num_heads, context_len, casual_attention }
    }

    pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let mut outputs = Vec::with_capacity(self.num_heads);
        for head in &self.casual_attention {
            let output = head.forward(input.clone());
            outputs.push(output);
        }
        Tensor::cat(outputs, 1)
    }
}

pub struct MultiHeadAttention<B: Backend> {
    d_out: usize,
    num_heads: usize,
    head_dim: usize,
    w_query: Linear<B>,
    w_key: Linear<B>,
    w_value: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(d_in: usize, d_out: usize, dropout: f64, num_heads: usize, qkv_bias: bool, device: B::Device) -> Self {
        assert!(d_out % num_heads == 0, "d_out must be divisible by num_heads");
        let head_dim = d_out / num_heads;
        let w_query = LinearConfig::new(d_in, d_out).with_bias(qkv_bias).init(&device);
        let w_key = LinearConfig::new(d_in, d_out).with_bias(qkv_bias).init(&device);
        let w_value = LinearConfig::new(d_in, d_out).with_bias(qkv_bias).init(&device);
        let out_proj = LinearConfig::new(d_out, d_out).with_bias(qkv_bias).init(&device);
        let dropout = DropoutConfig::new(dropout).init();
        Self {
            d_out,
            num_heads,
            head_dim,
            w_query,
            w_key,
            w_value,
            out_proj,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        let shape = input.shape();
        let batch_size = shape[0];
        let num_tokens = shape[1];
        let d_in = shape[2];
        let keys = self.w_key.forward(input.clone()).reshape([batch_size, num_tokens, self.num_heads, self.head_dim]);
        let queries = self.w_query.forward(input.clone()).reshape([batch_size, num_tokens, self.num_heads, self.head_dim]);
        let values = self.w_value.forward(input).reshape([batch_size, num_tokens, self.num_heads, self.head_dim]);
        let keys = keys.permute([0, 2, 1, 3]);
        let queries = queries.permute([0, 2, 1, 3]);
        let values = values.permute([0, 2, 1, 3]);
        // Perform scaled dot-product attention
        let attn_scores = queries.matmul(keys.permute([0, 1, 3, 2]));
        let score_mask = Tensor::tril_mask(attn_scores.shape(), 0, &attn_scores.device());
        let masked_scores = Tensor::mask_fill(attn_scores, score_mask, f32::NEG_INFINITY);
        let softmax_dim = masked_scores.dims().len() - 1;
        let attn_weights = softmax(masked_scores / (self.head_dim as f32).sqrt(), softmax_dim);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vec = attn_weights.matmul(values);
        let context_vec = context_vec.permute([0, 2, 1, 3]).reshape([batch_size, num_tokens, self.d_out]);
        let context_vec = self.out_proj.forward(context_vec);
        context_vec
    }
}
