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


#[derive(Debug)]
struct DummyGptBlock<B: Backend> {
    
}


impl<B: Backend> DummyGptBlock<B> {
    pub fn new(emb_dim: usize, n_heads: usize, drop_rate: f64, qkv_bias: bool) -> Self {
        DummyGptBlock {}
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
            DummyGptBlock::<B>::new(model_config.emb_dim, 
                model_config.n_heads, 
                model_config.drop_rate, 
                model_config.qkv_bias)).collect();
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
}