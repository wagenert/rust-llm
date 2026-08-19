use burn::module::Module;
use burn::prelude::*;
use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::activation::softmax,
};

#[derive(Module, Debug)]
pub struct CasualAttention<B: Backend> {
    w_query: Linear<B>,
    w_key: Linear<B>,
    w_value: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> CasualAttention<B> {
    pub fn new(d_in: usize, d_out: usize, dropout_rate: f64, qkv_bias: bool, device: &B::Device) -> Self {
        let linear_config = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let w_query = linear_config.clone().init(device);
        let w_key = linear_config.clone().init(device);
        let w_value = linear_config.init(device);
        let dropout = DropoutConfig::new(dropout_rate).init();

        Self {
            w_query,
            w_key,
            w_value,
            dropout,
            //device: device,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let query = self.w_query.forward(input.clone());
        let key = self.w_key.forward(input.clone());
        let attn_scores = query.clone().matmul(key.clone().transpose());

        // Create a lower triangular mask to prevent attention to future tokens.
        // Fill with negative infinity to ensure they don't contribute to the softmax.
        let score_mask = Tensor::tril_mask(attn_scores.shape(), 0, &attn_scores.device());
        let masked_scores = Tensor::mask_fill(attn_scores, score_mask, f32::NEG_INFINITY);
        let d_k = *key.shape().last().unwrap();
        let softmax_dim = masked_scores.dims().len() - 1;
        let attn_weights = softmax(masked_scores / (d_k as f32).sqrt(), softmax_dim);
        let attn_weights = self.dropout.forward(attn_weights);

        let value = self.w_value.forward(input.clone());
        attn_weights.matmul(value)
    }
}
