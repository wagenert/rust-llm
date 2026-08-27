use burn::backend::wgpu::Wgpu;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamW, GradientsParams, SimpleOptimizer};
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, BackendTypes};
use burn::{backend::Autodiff, data::dataloader::DataLoader};
use burn_helpers::{GptBatch, GptConfig, GptModel, create_dataloader};
use std::{fs, sync::Arc};
use tiktoken::CoreBpe;

type B = Wgpu<f32, i32>;
type OptimizeBackend = Autodiff<B>;

const FILEPATH: &str = "data/The_Verdict.txt";

fn calc_loss_batch<B: Backend>(batch: GptBatch<B>, model: &GptModel<B>, device: &B::Device) -> Tensor<B, 1> {
    let logits = model.forward(batch.input_ids);
    let loss = CrossEntropyLossConfig::new().init(device);

    loss.forward(logits.flatten(0, 1), batch.target_ids.flatten(0, 1))
}

fn calc_loss_loader<B: Backend>(
    data_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    model: &GptModel<B>,
    device: &B::Device,
    num_batches: Option<u32>,
) -> f32 {
    let mut total_loss = 0.0;
    let mut batches_count = 0;
    for batch in data_loader.iter() {
        batches_count += 1;
        let loss = calc_loss_batch(batch, model, device);
        total_loss += loss.mean().into_scalar().to_f32();
        if let Some(max_batches) = num_batches {
            if batches_count >= max_batches {
                break;
            }
        }
    }

    if batches_count == 0 {
        return f32::NAN;
    }
    total_loss / (batches_count as f32)
}

fn train_model_simple<B: AutodiffBackend>(
    model: &GptModel<B>,
    train_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    val_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    optimizer: AdamW,
    device: &B::Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &str,
    tokenizer: &CoreBpe,
) {
    let mut train_losses = Vec::new();
    let mut val_losses = Vec::new();
    let mut track_tokens_seen = Vec::new();
    let mut global_step = -1;
    let mut tokens_seen = 0;

    for epoch in 0..num_epochs {
        for batch in train_loader.iter() {
            //optimizer.zero_grad();
            let loss = calc_loss_batch(batch, model, device);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, model);
            let model = optimizer.step(0.0001, model, grads_params, None);
        }
    }
}

fn main() {
    let text = fs::read_to_string(FILEPATH).expect("Can not read file content");
    let total_characters = text.len();
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Can not initialize tokenizer");
    let total_tokens = tokenizer.encode(&text);
    println!("Total characters: {}", total_characters);
    println!("Total tokens: {}", total_tokens.len());
    let device = <OptimizeBackend as BackendTypes>::Device::default();
    OptimizeBackend::seed(&device, 123);

    let config = GptConfig::new().with_context_length(256);

    let train_ratio = 0.9;
    let split_idx = (train_ratio * text.len() as f64) as usize;
    let train_data = &text[..split_idx];
    let val_data = &text[split_idx..];
    let train_loader: Arc<dyn DataLoader<_, GptBatch<_>>> = create_dataloader::<OptimizeBackend>(
        train_data,
        2,
        config.context_length,
        config.context_length,
        true,
        0,
        &device,
    );
    let val_loader: Arc<dyn DataLoader<_, GptBatch<_>>> = create_dataloader::<OptimizeBackend>(
        val_data,
        2,
        config.context_length,
        config.context_length,
        true,
        0,
        &device,
    );
    let model = config.init(&device).no_grad();

    let train_loss = calc_loss_loader(train_loader, &model, &device, None);
    println!("Training loss: {}", train_loss);
    let val_loss = calc_loss_loader(val_loader, &model, &device, None);
    println!("Validation loss: {}", val_loss);
}
