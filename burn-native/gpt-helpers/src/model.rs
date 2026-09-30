use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::modules::transformer::TransformerEncoderConfig;
use burn::nn::transformer::TransformerEncoderInput;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};

use crate::NativeGptBatch;
use crate::transformer::{TransformerBlock, TransformerBlockConfig};

#[derive(Config, Debug)]
pub struct Gpt2ModelConfig {
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 1024)]
    pub context_length: usize,
    #[config(default = 768)]
    pub n_embed: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = 12)]
    pub n_layers: usize,
    #[config(default = 0.1)]
    pub drop_rate: f64,
    #[config(default = false)]
    pub qkv_bias: bool,
    #[config(default = 1e-5)]
    pub layer_norm_epsilon: f64,
    #[config(default = 1024)]
    pub max_seq_len: usize,
}

impl Gpt2ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2Model<B> {
        Gpt2Model::new(self, device)
    }
}

#[derive(Debug, Module)]
pub struct Gpt2Model<B: Backend> {
    wte: Embedding<B>,
    wpe: Embedding<B>,
    dropout: Dropout,
    transformers: Vec<TransformerBlock<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> Gpt2Model<B> {
    pub fn new(config: &Gpt2ModelConfig, device: &B::Device) -> Self {
        let wte = EmbeddingConfig::new(config.vocab_size, config.emb_dim).init(device);
        let wpe = EmbeddingConfig::new(config.context_length, config.emb_dim).init(device);
        let dropout = DropoutConfig::new(config.drop_rate).init();

        let mut transformers = Vec::with_capacity(config.n_heads);
        for _i in 0..config.n_heads {
            let tranformer = TransformerBlockConfig::new(config).init(device);
            transformers.push(tranformer);
        }
        let ln_f = LayerNormConfig::new(config.n_embed).init(device);
        let lm_head = LinearConfig::new(config.n_embed, config.vocab_size).init(device);

        Self {
            wte,
            wpe,
            dropout,
            transformers,
            ln_f,
            lm_head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let input_shape = input.shape();
        let _batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let tok_embeds = self.wte.forward(input.clone());
        let pos_input =
            Tensor::<B, 1, Int>::from_data(Vec::from_iter(0..seq_length).as_slice(), &input.device()).unsqueeze();
        let pos_embeds = self.wpe.forward(pos_input);
        let x = tok_embeds + pos_embeds;
        let x = self.dropout.forward(x);
        let x = TransformerEncoderInput::new(x);
        let x = self.transformers.forward(x);
        let x = self.ln_f.forward(x);
        let x = self.lm_head.forward(x);
        x.flatten(0, 1)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let flat_targets = targets.clone().flatten(0, 1);
        let output = self.forward(input);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), flat_targets.clone());
        ClassificationOutput::new(loss, output, flat_targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Gpt2Model<B> {
    type Input = NativeGptBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, input: Self::Input) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(input.input_ids, input.target_ids);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Gpt2Model<B> {
    type Input = NativeGptBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, input: Self::Input) -> Self::Output {
        self.forward_classification(input.input_ids, input.target_ids)
    }
}
