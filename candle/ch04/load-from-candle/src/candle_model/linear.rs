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
        let (batch, seq_len, in_features) = x.dims3()?;
        let x = x.reshape((batch * seq_len, in_features))?;
        let y = x.matmul(&self.weights)?.broadcast_add(&self.bias)?;
        let out_features = self.weights.dim(1)?;
        y.reshape((batch, seq_len, out_features))
    }
}