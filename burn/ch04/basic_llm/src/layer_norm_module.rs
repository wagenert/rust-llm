use burn::{Tensor, tensor::backend::Backend};
use burn::module::Module;


#[derive(Module, Debug)]
pub struct LayerNormModule<B: Backend> {
    eps: f64,
    scale: Tensor<B, 2>,
    shift: Tensor<B, 2>,
}

impl<B: Backend> LayerNormModule<B> {
    pub fn new(eps: f64, emb_dim: usize, device: B::Device) -> Self {
        let scale = Tensor::<B, 2>::zeros([emb_dim], &device);
        let shift = Tensor::<B, 2>::zeros([emb_dim], &device);
        Self { eps, scale, shift }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = input.clone().mean_dim(1);
        let variance = input.clone().var(1);
        let normalized = (input - mean) / (variance + self.eps).sqrt();
        self.scale.clone().matmul(normalized).add(self.shift.clone())
    }
}