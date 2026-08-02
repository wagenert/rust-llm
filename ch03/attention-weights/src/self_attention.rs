use burn::{Tensor, nn::{Linear, LinearConfig}, prelude::Backend, tensor::{Float, activation::softmax}};

pub struct SelfAttention<B: Backend> {
    w_query: Linear<B>,
    w_key: Linear<B>,
    w_value: Linear<B>,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool, device: &B::Device) -> Self {
        let linear_config = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let w_query = linear_config.clone().init(device);
        let w_key = linear_config.clone().init(device);
        let w_value = linear_config.init(device);

        Self {
            w_query,
            w_key,
            w_value,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let query = self.w_query.forward(input.clone());
        let key = self.w_key.forward(input.clone());
        let value = self.w_value.forward(input.clone());

        let attn_scores = query.clone().matmul(key.clone().transpose());
        let d_k = *key.shape().last().unwrap_or(&1);
        let attn_weights = softmax(attn_scores / (d_k as f32).sqrt(), 1);

        attn_weights.matmul(value)
    }

}
