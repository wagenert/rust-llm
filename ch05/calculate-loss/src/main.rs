use burn::backend::flex::Flex;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn::{backend::Autodiff, data::dataloader::DataLoader};
use burn_helpers::{GptBatch, GptConfig, create_dataloader};
use std::{fs, sync::Arc};

type B = Flex<f32, i32>;
type OptimizeBackend = Autodiff<B>;

const FILEPATH: &str = "data/The_Verdict.txt";

fn main() {
    let text = fs::read_to_string(FILEPATH).expect("Can not read file content");
    let total_characters = text.len();
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Can not initialize tokenizer");
    let total_tokens = tokenizer.encode(&text);
    println!("Total characters: {}", total_characters);
    println!("Total tokens: {}", total_tokens.len());
    let device = <OptimizeBackend as BackendTypes>::Device::default();
    let config = GptConfig::new();
    let train_ratio = 0.9;
    let split_idx = (train_ratio * text.len() as f64) as usize;
    let train_data = &text[..split_idx];
    let val_data = &text[split_idx..];
    let train_loader: Arc<dyn DataLoader<_, GptBatch<OptimizeBackend>>> = create_dataloader(
        train_data,
        2,
        config.context_length,
        config.context_length,
        true,
        0,
        &device,
    );
    let val_loader: Arc<dyn DataLoader<_, GptBatch<OptimizeBackend>>> = create_dataloader(
        val_data,
        2,
        config.context_length,
        config.context_length,
        true,
        0,
        &device,
    );
    println!("Training batches:");
    for batch in train_loader.iter() {
        println!("{} {}", batch.input_ids.shape(), batch.target_ids);
    }

    println!("Validation batches:");
    for batch in val_loader.iter() {
        println!("{} {}", batch.input_ids.shape(), batch.target_ids);
    }

    OptimizeBackend::seed(&device, 123);
}
