use crate::gpt_config::GptConfig124M;
use crate::transformer_block::TransformerBlock;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct GptModel<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    dropout: Dropout,
    transformer_block: Vec<TransformerBlock<B>>,
    final_norm: LayerNorm<B>,
    out_head: Linear<B>,
}

impl<B: Backend> GptModel<B> {
    pub fn new(cfg: &GptConfig124M, device: B::Device) -> Self {
        let tok_emb = EmbeddingConfig::new(cfg.vocab_size, cfg.emb_dim).init(&device);
        let pos_emb = EmbeddingConfig::new(cfg.context_length, cfg.emb_dim).init(&device);
        let dropout = DropoutConfig::new(cfg.drop_rate).init();
        let mut transformer_block = Vec::<TransformerBlock<B>>::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            transformer_block.push(TransformerBlock::new(&cfg, device.clone()))
        }
        let final_norm = LayerNormConfig::new(cfg.emb_dim).init(&device);
        let out_head = LinearConfig::new(cfg.emb_dim, cfg.vocab_size).init(&device);
        Self {
            tok_emb,
            pos_emb,
            dropout,
            transformer_block,
            final_norm,
            out_head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Implement the forward pass of the model
        // For example, process the input and return the output
        let input_shape = input.shape();
        let _batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let tok_embeds = self.tok_emb.forward(input.clone());
        let pos_input =
            Tensor::<B, 1, Int>::from_data(Vec::from_iter(0..seq_length).as_slice(), &input.device()).unsqueeze();
        let pos_embeds = self.pos_emb.forward(pos_input);
        let x = tok_embeds + pos_embeds;
        let x = self.dropout.forward(x);
        let x = self.transformer_block.iter().fold(x, |x, tf| tf.forward(x));
        let x = self.final_norm.forward(x);
        self.out_head.forward(x)
    }
}
