use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use native_gpt::BurnModelConfig;

type InnerBackend = Wgpu<f32, i32>;
type OptimizerBackend = Autodiff<InnerBackend>;

const FILENAME: &str = "data/The_Verdict.txt";
fn main() {
    let file = std::fs::read_to_string(FILENAME).expect("Failed to read file");
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Unable to initiatlize tokenizer");
    let _input_tokens = tokenizer.encode(&file);
    let device = <OptimizerBackend as BackendTypes>::Device::default();
    OptimizerBackend::seed(&device, 123);
    let _model = BurnModelConfig::new().init::<OptimizerBackend>(&device);
}
