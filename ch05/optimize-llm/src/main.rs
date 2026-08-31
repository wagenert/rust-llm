use burn::backend::Flex;
use burn::backend::wgpu::Wgpu;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::backend::{AutodiffBackend, BackendTypes};
use burn::{backend::Autodiff, data::dataloader::DataLoader};
use burn_helpers::{GptBatch, GptConfig, GptModel, create_dataloader};
use std::{fs, sync::Arc};
use tiktoken::CoreBpe;

type InnerBackend = Flex<f32, i32>;
type OptimizeBackend = Autodiff<InnerBackend>;

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
    model: &mut GptModel<B>,
    train_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    val_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    optimizer: OptimizerAdaptor<AdamW, GptModel<B>, B>,
    device: &B::Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &str,
    tokenizer: &CoreBpe,
) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut model = GptModel::<B>::clone(model);
    let mut optimizer = optimizer.clone();
    let mut train_losses = Vec::new();
    let mut val_losses = Vec::new();
    let mut track_tokens_seen = Vec::new();
    let mut global_step = -1;
    let mut tokens_seen = 0;

    for epoch in 0..num_epochs {
        for batch in Arc::clone(&train_loader).iter() {
            let loss = calc_loss_batch(batch.clone(), &model, device);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(0.0001, model, grads_params);

            tokens_seen += batch.input_ids.num_params();
            global_step += 1;
            let (train_loss, val_loss) = evaluate_model(
                &model,
                Arc::clone(&train_loader),
                Arc::clone(&val_loader),
                device,
                eval_iter,
            );
            train_losses.push(train_loss);
            val_losses.push(val_loss);
            track_tokens_seen.push(tokens_seen);
            println!(
                "Ep {} (Step {:06}): Train loss {:.3}, Val loss {:.3}",
                epoch + 1,
                global_step,
                train_loss,
                val_loss
            );
        }
        generate_and_print_sample(&model, tokenizer, device, start_context);
    }
    (train_losses, val_losses, track_tokens_seen)
}

fn evaluate_model<B: AutodiffBackend>(
    model: &GptModel<B>,
    train_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    val_loader: Arc<dyn DataLoader<B, GptBatch<B>>>,
    device: &B::Device,
    eval_iter: usize,
) -> (f32, f32) {
    let train_loss = calc_loss_loader(train_loader, &model, device, Some(eval_iter as u32));
    let val_loss = calc_loss_loader(val_loader, &model, device, Some(eval_iter as u32));
    (train_loss, val_loss)
}

fn generate_and_print_sample<B: AutodiffBackend>(
    model: &GptModel<B>,
    tokenizer: &CoreBpe,
    device: &B::Device,
    start_context: &str,
) {
    let eval_model = model.valid();
    let input_ids = tokenizer.encode(start_context);
    let input_tensor: burn::tensor::Tensor<B::InnerBackend, 2, Int> =
        burn::tensor::Tensor::<B::InnerBackend, 1, Int>::from_data(input_ids.as_slice(), device).unsqueeze_dim::<2>(0);
    let output = eval_model.forward(input_tensor);
    // println!("Output shape {:?}", output.shape());
    let generated_ids = output.clone().argmax(output.dims().len() - 1);
    let generated_text = tokenizer.decode(&generated_ids.to_data().convert_dtype(DType::U32).as_slice().unwrap());
    println!(
        "Generated sample: {} {}",
        start_context,
        String::from_utf8(generated_text)
            .unwrap()
            .replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
    );
}

fn main() {
    let text = fs::read_to_string(FILEPATH).expect("Can not read file content");
    // let total_characters = text.len();
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Can not initialize tokenizer");
    // let total_tokens = tokenizer.encode(&text);
    // println!("Total characters: {}", total_characters);
    // println!("Total tokens: {}", total_tokens.len());
    let device = <OptimizeBackend as BackendTypes>::Device::default();
    OptimizeBackend::seed(&device, 123);

    let config = GptConfig::new().with_context_length(256);

    let train_ratio = 0.9;
    let split_idx = (train_ratio * text.len() as f64) as usize;
    let train_data = &text[..split_idx];
    let val_data = &text[split_idx..];
    let train_loader: Arc<dyn DataLoader<_, GptBatch<_>>> = create_dataloader::<OptimizeBackend>(
        train_data,
        tokenizer,
        2,
        config.context_length,
        config.context_length,
        true,
        0,
        &device,
    );
    let val_loader: Arc<dyn DataLoader<_, GptBatch<_>>> = create_dataloader::<OptimizeBackend>(
        val_data,
        tokenizer,
        2,
        config.context_length,
        config.context_length,
        true,
        0,
        &device,
    );
    let mut model = config.init(&device);
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.1)
        .init::<OptimizeBackend, GptModel<OptimizeBackend>>();

    let num_epochs = 10;
    let eval_freq = 5;
    let eval_iter = 5;
    let start_context = "Every effort moves you";
    let (train_losses, val_losses, tokens_seen) = train_model_simple(
        &mut model,
        train_loader,
        val_loader,
        optimizer,
        &device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
    );
    /*
    let train_loss = calc_loss_loader(train_loader, &model, &device, None);
    println!("Training loss: {}", train_loss);
    let val_loss = calc_loss_loader(val_loader, &model, &device, None);
    println!("Validation loss: {}", val_loss);
    */
}
