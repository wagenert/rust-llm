use burn::{backend::Wgpu, prelude::*, record::CompactRecorder, tensor::backend::BackendTypes};
use gpt_helpers::{Gpt2Model, Gpt2ModelConfig, generate_text_simple};
use infer_gpt2::text_token_converter;

type B = Wgpu<f32, i32>;

const MODEL_PATH: &str = "artifacts-stored/model.mpk";
fn main() {
    let input_text = "A small step for me, but a giant leap for";
    let device = <B as BackendTypes>::Device::default();
    let model: Gpt2Model<B> = Gpt2ModelConfig::new().with_qkv_bias(true).init(&device);
    let model = model.load_file(MODEL_PATH, &CompactRecorder::new(), &device).unwrap();
    let model= model.no_grad();
    println!("Model initialized successfully!");
    let text_token_converter = text_token_converter::TextTokenConverter::new("gpt2");
    let input_ids = text_token_converter.text_to_token_ids(input_text, &device);
    let output_ids = generate_text_simple(&model, input_ids, 50, model.get_context_length() as i32, None, None, None); 
    let output = text_token_converter.token_ids_to_text(output_ids).unwrap();
    println!("Next text: {:?}", output);
}
