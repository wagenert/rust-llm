use::burn::module::Module;
use burn::{nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::*};
pub struct GptConfig124M {
    vocab_size: usize,
    context_length: usize,
    emb_dim: usize,
    n_heads: usize,
    n_layers: usize,
    drop_rate: f64,
    qkv_bias: bool,
}


#[derive(Module, Debug)]
struct DummyGptBlock<B: Backend> {
    eps: f64,
    _phantom: std::marker::PhantomData<B>,
}


impl<B: Backend> DummyGptBlock<B> {
    pub fn new(_cfg: &GptConfig124M, eps: f64) -> Self {
        Self { eps, _phantom: std::marker::PhantomData }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        input
    }

}

#[derive(Module, Debug)]
pub struct GptDummyModel<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    drop_emb: Dropout,
    trf_blocks: Vec<DummyGptBlock<B>>,
    norm: LayerNorm<B>,
    out_head: Linear<B>,
}

impl<B: Backend> GptDummyModel<B> {
    pub fn new(model_config: &GptConfig124M, device: B::Device) -> Self {
        let tok_emb = EmbeddingConfig::new(model_config.vocab_size, model_config.emb_dim).init(&device);
        let pos_emb = EmbeddingConfig::new(model_config.context_length, model_config.emb_dim).init(&device);
        let drop_emb = DropoutConfig::new(model_config.drop_rate).init();
        let trf_blocks = (0..model_config.n_layers).map(|_| 
            DummyGptBlock::<B>::new(model_config, 1e-5)).collect();
        let norm = LayerNormConfig::new(model_config.emb_dim).init(&device);
        let out_head = LinearConfig::new(model_config.emb_dim, model_config.vocab_size).init(&device);

        GptDummyModel {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            norm,
            out_head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input.shape()[..2].try_into().unwrap();
        let tok_emb = self.tok_emb.forward(input.clone());
        let pos_ids = Tensor::<B, 1, Int>::from_data(Vec::from_iter(0..seq_len as i64).as_slice(), &input.device()).unsqueeze::<2>();
        let pos_emb: Tensor<B, 3> = self.pos_emb.forward(pos_ids);

        let mut x = tok_emb + pos_emb;
        x = self.drop_emb.forward(x);

        for block in &self.trf_blocks {
            x = block.forward(x);
        }

        x = self.norm.forward(x);
        let logits = self.out_head.forward(x);
        logits
    }
}
