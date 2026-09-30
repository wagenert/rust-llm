use candle_core::Tensor;
use candle_nn::VarBuilder;

pub struct Gpt2Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Gpt2Linear {
    pub fn new(indim: usize, outdim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weights = vb.get((indim, outdim), "weight")?;
        let bias = vb.get(outdim, "bias")?;
        let model = Self {
            weights,
            bias,
        };
        Ok(model)
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        //let w = self.weights.t()?;
        let w = &self.weights;
        x.matmul(&w)?.broadcast_add(&self.bias)
    }
}