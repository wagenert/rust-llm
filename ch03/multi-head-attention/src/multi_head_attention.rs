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
