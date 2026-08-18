use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::prelude::*;
use burn_helpers::attention_algorithms::multi_head_attention::MultiHeadAttention;
use llm_helpers::feed_forward_module::FeedForwardModule;
use llm_helpers::gpt_config::GptConfig124M;

pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    feedforward: FeedForwardModule<B>,
    layer_norm1: LayerNorm<B>,
    layer_norm2: LayerNorm<B>,
    drop_shortcut: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: GptConfig124M, device: B::Device) -> Self {
        let attention = MultiHeadAttention::<B>::new(
            config.emb_dim,
            config.emb_dim,
            config.context_length,
            config.drop_rate,
            config.n_heads,
            config.qkv_bias,
            device.clone(),
        );
        let feedforward = FeedForwardModule::<B>::new(&config, &device);
        let layer_norm1 = LayerNormConfig::new(config.emb_dim).init(&device);
        let layer_norm2 = LayerNormConfig::new(config.emb_dim).init(&device);
        let drop_shortcut = DropoutConfig::new(config.drop_rate).init();
        Self {
            attention,
            feedforward,
            layer_norm1,
            layer_norm2,
            drop_shortcut,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.layer_norm1.forward(input.clone());
        let x = self.attention.forward(x);
        let x = self.drop_shortcut.forward(x);
        let x = input.clone() + x;

        let x = self.layer_norm2.forward(x);
        let x = self.feedforward.forward(x);
        let x = self.drop_shortcut.forward(x);
        input + x
    }
}
