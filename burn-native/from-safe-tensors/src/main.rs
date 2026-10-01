use burn::{backend::Wgpu, module::Module, record::CompactRecorder, tensor::backend::BackendTypes};
use gpt_helpers::{Gpt2Model, Gpt2ModelConfig};
use from_safe_tensors::tensormap;


type B = Wgpu<f32, i32>;

const MODEL_FILE: &str = "/Users/thomaswagener/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
const MODEL_PATH: &str = "artifacts_gpt2_model.burn";
fn main() {
    let device = <B as BackendTypes>::Device::default();
    let tensor_map = tensormap::load_tensor_map(MODEL_FILE).unwrap();
    println!("Tensor map loaded successfully!");
    let model: Gpt2Model<B> = Gpt2ModelConfig::new().with_qkv_bias(true).init(&device);
    let model = tensormap::load_gpt2_weights(model, &tensor_map, &device).unwrap();
    println!("Model weights loaded successfully!");
    model.save_file(MODEL_PATH, &CompactRecorder::new()).unwrap();

}
