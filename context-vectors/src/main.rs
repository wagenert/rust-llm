use burn::backend::Wgpu;
use burn_helpers::gpt_dataset::{GPTBatch, create_dataloader};
use burn::nn::Embedding;
use burn::data::dataloader::DataLoader;
use std::sync::Arc;

type MyBackend = Wgpu<f32, i32>;

const FILE_PATH: &str = "../data/The_Verdict.txt";

fn main() {
    let file = std::fs::File::open(FILE_PATH).expect("Failed to open file");
    let txt = std::io::read_to_string(file).expect("Failed to read file");
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Failed to get encoding");

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
    let vocab_size = 50257;
    let output_dim = 256;
    let token_embedding_layer: Embedding<_> = burn::nn::EmbeddingConfig::new(vocab_size, output_dim)
        .init::<Wgpu>(&device);

    let max_length = 4;
    let batch_size = 8;
    let stride = max_length;
    let shuffle = false;
    let num_workers = 1;
    let dataloader: Arc<dyn DataLoader<MyBackend, GPTBatch<MyBackend>>> = create_dataloader(&txt, batch_size, max_length, stride, shuffle, num_workers, device.clone());
    let mut data_iter = dataloader.iter();
    let first_batch = data_iter.next().expect("Failed to get first batch");
    let token_embeddings =token_embedding_layer.forward(first_batch.input_ids.clone());
    println!("Shape of embedded input_ids: {:?}", token_embeddings.shape());

}
