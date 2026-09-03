use std::sync::Arc;

use crate::NativeGptBatch;
use crate::dataset::NativeGptDataset;
use crate::{NativeGptDataBatcher, model::BurnModelConfig};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};
use burn::{config::Config, data::dataloader::DataLoaderBuilder, optim::AdamWConfig, tensor::backend::AutodiffBackend};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: BurnModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 0.0001)]
    pub learning_rate: f64,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 10)]
    pub epochs: usize,
    #[config(default = 42)]
    pub seed: u64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).expect("Failed to create artifact directory");
}
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    input: &str,
    tokenizer: &tiktoken::CoreBpe,
    config: TrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{}/config.json", artifact_dir))
        .expect("Failed to save config");

    B::seed(&device, config.seed);

    let batcher = NativeGptDataBatcher::default();
    let dataset = NativeGptDataset::new(input, tokenizer, 768, 64);

    let dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<B, NativeGptBatch<B>>> =
        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .shuffle(config.seed)
            .build(dataset.train());

    let dataloader_test: Arc<dyn burn::data::dataloader::DataLoader<B::InnerBackend, NativeGptBatch<B::InnerBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .shuffle(config.seed)
            .build(dataset.test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(), AccuracyMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(model, config.optimizer.init(), config.learning_rate));

    result
        .model
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Failed to save model");
}
