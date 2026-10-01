use burn::{backend::Wgpu, prelude::*, record::CompactRecorder, tensor::{DType, backend::BackendTypes}};
use gpt_helpers::{Gpt2Model, Gpt2ModelConfig};

type B = Wgpu<f32, i32>;

const MODEL_PATH: &str = "artifacts-stored/model.mpk";
fn main() {
    let input_text = "Every step moves you";
    let device = <B as BackendTypes>::Device::default();
    let model: Gpt2Model<B> = Gpt2ModelConfig::new().with_qkv_bias(true).init(&device);
    let model = model.load_file(MODEL_PATH, &CompactRecorder::new(), &device).unwrap();
    println!("Model initialized successfully!");
    let tokenizer = tiktoken::get_encoding("gpt2").unwrap();
    let token_ids = tokenizer.encode_with_special_tokens(input_text);
    let input_ids = Tensor::<B, 1, Int>::from_data(token_ids.as_slice(), &device).unsqueeze_dim::<2>(0);

    let logits = model.forward(input_ids);
    // let logits = logits.slice(s![ -1, ..]);
    println!("Output tensor shape: {:?}", logits.shape());    
    let probas = burn::tensor::activation::softmax(logits, 1);
    println!("Probas tensor shape: {:?}", probas);
    let idx_next: Tensor<B, 1, Int> = probas.argmax(1).squeeze_dim(1);
    println!("Next token id: {:?}", idx_next);
    let token_next = idx_next.to_data().to_vec::<i32>().unwrap();
    println!("Next token id: {:?}", token_next);    

    let next_text_bytes = tiktoken::get_encoding("gpt2").unwrap().decode(&[1164]);
    let next_text = String::from_utf8(next_text_bytes).expect("Can not convert to string");
    println!("Next text: {:?}", next_text);
}
