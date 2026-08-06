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
    
}


impl<B: Backend> DummyGptBlock<B> {
    pub fn new(cfg: &GptConfig124M, eps: f64) -> Self {
        Self {}
    }
}

impl Module for DummyGptBlock<B> {
    type Input = Tensor<B, 3>;
    type Output = Tensor<B, 3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
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

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
        let seq_len = input.dims().1;
        let tok_emb = self.tok_emb.forward(input);
        let pos_ids = Tensor::<B, 2>::arange(seq_len as i64, (1, seq_len as i64), &input.device());
        let pos_emb = self.pos_emb.forward(pos_ids);
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
