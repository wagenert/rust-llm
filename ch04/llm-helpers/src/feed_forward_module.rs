use burn::module::Module;
use burn::{
    nn::{Gelu, Linear, LinearConfig},
    tensor::backend::Backend,
    Tensor,
};

use crate::gpt_config::GptConfig124M;

#[derive(Module, Debug)]
pub struct FeedForwardModule<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub activation: Gelu,
}

impl<B: Backend> FeedForwardModule<B> {
    pub fn new(cfg: &GptConfig124M, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(cfg.emb_dim, 4 * cfg.emb_dim).init(device);
        let linear2 = LinearConfig::new(4 * cfg.emb_dim, cfg.emb_dim).init(device);
        let activation = Gelu::new();
        Self {
            linear1,
            linear2,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}
