use std::sync::Arc;

use burn::{Tensor, backend::Wgpu, data::dataloader::DataLoader, nn::Embedding, tensor::Int};

use burn_helpers::{GptBatch, create_dataloader};

type MyBackend = Wgpu<f32, i32>;

const FILE_PATH: &str = "../data/The_Verdict.txt";
fn main() {
    let file = std::fs::File::open(FILE_PATH).expect("Failed to open file");
    let txt = std::io::read_to_string(file).expect("Failed to read file");
    let tokenizer: &tiktoken::CoreBpe = tiktoken::get_encoding("gpt2").expect("Failed to get encoding");

    let enc_text = tokenizer.encode(&txt);
    println!("Encoded text length: {:?}", enc_text.len());
    let enc_sample = enc_text[..50].to_vec();
    println!("Encoded sample: {:?}", enc_sample);
    let context_size = 4;
    let x = enc_sample[..context_size].to_vec();
    println!("Context: {:?}", x);
    let y = enc_sample[1..context_size + 1].to_vec();
    println!("Next token: {:?}", y);

    for i in 1..context_size + 1 {
        let context = enc_sample[..i].to_vec();
        let desired = enc_sample[i];
        println!("Context: {:?}, Next token: {:?}", context, desired);
    }

    let device = burn::backend::wgpu::WgpuDevice::default();
    /*
    let batch_size = 8;
    let max_length = 4;
    let stride = 4;
    let shuffle = false;
    let num_workers = 1;
    let dataloader: Arc<dyn DataLoader<MyBackend, GPTBatch<MyBackend>>> = create_dataloader(&txt, batch_size, max_length, stride, shuffle, num_workers, device.clone());
    let mut data_iter = dataloader.iter();
    let first_batch = data_iter.next().expect("Failed to get first batch");
    println!("First batch input_ids: {}", first_batch.input_ids);
    println!("First batch target_ids: {}", first_batch.target_ids);
    let second_batch = data_iter.next().expect("Failed to get second batch");
    println!("Second batch input_ids: {}", second_batch.input_ids.to_data());
    println!("Second batch target_ids: {}", second_batch.target_ids.to_data());

    let input_ids = Tensor::<MyBackend, 1, Int>::from_data([2, 3, 5, 1], &device).reshape([1, 4]);
    let vocab_size = 6;
    let output_dim = 3;

    let embedding_layer: Embedding<_> = burn::nn::EmbeddingConfig::new(vocab_size, output_dim)
        .init::<Wgpu>(&device);

    println!("Embedding for input_ids: {}", embedding_layer.weight);
    println!("Embedding for input_ids: {}", embedding_layer.forward(input_ids));
    */
    let vocab_size = 50257;
    let output_dim = 256;
    let token_embedding_layer: Embedding<_> =
        burn::nn::EmbeddingConfig::new(vocab_size, output_dim).init::<Wgpu>(&device);

    let max_length = 4;
    let batch_size = 8;
    let stride = max_length;
    let shuffle = false;
    let num_workers = 1;
    let dataloader: Arc<dyn DataLoader<MyBackend, GptBatch<MyBackend>>> = create_dataloader(
        &txt,
        tokenizer,
        batch_size,
        max_length,
        stride,
        shuffle,
        num_workers,
        &device,
    );
    let mut data_iter = dataloader.iter();
    let first_batch = data_iter.next().expect("Failed to get first batch");
    println!("First batch input_ids: {}", first_batch.input_ids);

    let token_embeddings = token_embedding_layer.forward(first_batch.input_ids.clone());
    println!("Shape of embedded input_ids: {:?}", token_embeddings.shape());

    let context_length = max_length;
    let pos_embedding_layer = burn::nn::EmbeddingConfig::new(context_length, output_dim).init::<Wgpu>(&device);
    let range: Vec<u32> = (0..(context_length as u32)).collect();
    let pos_embeddings = pos_embedding_layer
        .forward(Tensor::<MyBackend, 1, Int>::from_data(&range[..], &device).reshape([1, context_length]));
    println!("Shape of positional embeddings: {:?}", pos_embeddings.shape());
    let input_embeddings = token_embeddings + pos_embeddings;
    println!("Shape of input embeddings: {:?}", input_embeddings.shape());
}
