use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::Bool;
use burn::tensor::activation::softmax;
use burn_helpers::casual_attention::CasualAttention;
use burn::tensor::{backend::Backend, Tensor};
use burn::prelude::Float;

pub struct MultiHeadAttentionWrapper<B: Backend> {
    num_heads: usize,
    _context_len: usize,
    casual_attention: Vec<CasualAttention<B>>,
}

impl<B: Backend> MultiHeadAttentionWrapper<B> {
    pub fn new(d_in: usize, d_out: usize, context_len: usize, dropout: f64, num_heads: usize, qkv_bias: bool, device: B::Device) -> Self {
        let casual_attention = (0..num_heads).map(
            |_| CasualAttention::<B>::new(d_in, d_out, dropout, qkv_bias, &device)
        ).collect();
        Self { num_heads, _context_len: context_len, casual_attention }
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
    mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(d_in: usize, d_out: usize, context_length: usize, dropout: f64, num_heads: usize, qkv_bias: bool, device: B::Device) -> Self {
        assert!(d_out % num_heads == 0, "d_out must be divisible by num_heads");
        let head_dim = d_out / num_heads;
        let w_query = LinearConfig::new(d_in, d_out).with_bias(qkv_bias).init(&device);
        let w_key = LinearConfig::new(d_in, d_out).with_bias(qkv_bias).init(&device);
        let w_value = LinearConfig::new(d_in, d_out).with_bias(qkv_bias).init(&device);
        let out_proj = LinearConfig::new(d_out, d_out).with_bias(qkv_bias).init(&device);
        let dropout = DropoutConfig::new(dropout).init();
        let mask = Tensor::tril_mask([context_length, context_length], 0, &device);
        Self {
            d_out,
            num_heads,
            head_dim,
            w_query,
            w_key,
            w_value,
            out_proj,
            dropout,
            mask,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        let shape = input.shape();
        // println!("Input Shape: {:?}", shape);
        let batch_size = shape[0];
        let num_tokens = shape[1];
        let _d_in = shape[2];
        let keys = self.w_key.forward(input.clone())
            .reshape([batch_size, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let queries = self.w_query.forward(input.clone())
            .reshape([batch_size, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let values = self.w_value.forward(input)
            .reshape([batch_size, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Perform scaled dot-product attention
        let attn_scores = queries.matmul(keys.clone().swap_dims(2, 3));
        let score_mask = self.mask.clone()
            .slice([..num_tokens, ..num_tokens])
            .unsqueeze::<4>();
        let attn_scores = Tensor::mask_fill(attn_scores, score_mask, f32::NEG_INFINITY);
        let softmax_dim = attn_scores.dims().len() - 1;
        let key_dim = *keys.shape().last().unwrap();
        let attn_weights = softmax(attn_scores / (key_dim as f32).sqrt(), softmax_dim);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vec = attn_weights.matmul(values)
            .swap_dims(1, 2)
            .reshape([batch_size, num_tokens, self.d_out]);
        let context_vec = self.out_proj.forward(context_vec);
        context_vec
    }
}
