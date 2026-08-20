use burn::prelude::*;
use full_model::gpt_model::GptModel;
use llm_helpers::gpt_config::GptConfig124M;

type B = burn::backend::Flex<f32, i32>;
type ComputeBackend = burn::backend::Autodiff<B>;
fn main() {
    let config = GptConfig124M {
        vocab_size: 50257,
        context_length: 1024,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };
    let device = <ComputeBackend as burn::tensor::backend::BackendTypes>::Device::default();
    ComputeBackend::seed(&device, 123);
    let txt1 = "Every effort moves you";
    let txt2 = "Every day holds a";

    let tokenizer = tiktoken::get_encoding("gpt2").unwrap();
    let enc_txt1: Tensor<ComputeBackend, 1, Int> =
        Tensor::from_data(tokenizer.encode(txt1).as_slice(), &device.clone());
    let enc_txt2: Tensor<ComputeBackend, 1, Int> =
        Tensor::from_data(tokenizer.encode(txt2).as_slice(), &device.clone());
    let batch: Tensor<ComputeBackend, 2, Int> = Tensor::stack([enc_txt1, enc_txt2].to_vec(), 0);

    let model = GptModel::<ComputeBackend>::new(&config, device.clone());
    let output = model.forward(batch.clone());
    println!("Input shape: {:?}", batch.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Output: {:?}", output);
}
