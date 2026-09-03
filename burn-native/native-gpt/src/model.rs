use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::modules::transformer::{TransformerEncoder, TransformerEncoderConfig};
use burn::nn::transformer::TransformerEncoderInput;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};

use crate::NativeGptBatch;

#[derive(Config, Debug)]
pub struct BurnModelConfig {
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 1024)]
    pub context_length: usize,
    #[config(default = 768)]
    pub emb_dim: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = 12)]
    pub n_layers: usize,
    #[config(default = 0.1)]
    pub drop_rate: f64,
    #[config(default = false)]
    pub qkv_bias: bool,
}

impl BurnModelConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> BurnModel<B> {
        BurnModel::new(self, device)
    }
}

#[derive(Debug, Module)]
pub struct BurnModel<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    dropout: Dropout,
    transformers: TransformerEncoder<B>,
    final_norm: LayerNorm<B>,
    output_layer: Linear<B>,
}

impl<B: Backend> BurnModel<B> {
    pub fn new(config: BurnModelConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.emb_dim).init(device);
        let positional_embedding = EmbeddingConfig::new(config.context_length, config.emb_dim).init(device);
        let dropout = DropoutConfig::new(config.drop_rate).init();
        let transformers =
            TransformerEncoderConfig::new(config.emb_dim, config.emb_dim, config.n_heads, config.n_layers).init(device);
        let final_norm = LayerNormConfig::new(config.emb_dim).init(device);
        let output_layer = LinearConfig::new(config.emb_dim, config.vocab_size).init(device);

        Self {
            token_embedding,
            positional_embedding,
            dropout,
            transformers,
            final_norm,
            output_layer,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let input_shape = input.shape();
        let _batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let tok_embeds = self.token_embedding.forward(input.clone());
        let pos_input =
            Tensor::<B, 1, Int>::from_data(Vec::from_iter(0..seq_length).as_slice(), &input.device()).unsqueeze();
        let pos_embeds = self.positional_embedding.forward(pos_input);
        let x = tok_embeds + pos_embeds;
        let x = self.dropout.forward(x);
        let x = TransformerEncoderInput::new(x);
        let x = self.transformers.forward(x);
        let x = self.final_norm.forward(x);
        let x = self.output_layer.forward(x);
        x.flatten(0, 1)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(input);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for BurnModel<B> {
    type Input = NativeGptBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, input: Self::Input) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(input.input_ids, input.target_ids);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for BurnModel<B> {
    type Input = NativeGptBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, input: Self::Input) -> Self::Output {
        self.forward_classification(input.input_ids, input.target_ids)
    }
}
