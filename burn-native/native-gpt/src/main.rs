use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::optim::AdamWConfig;
use burn::tensor::backend::BackendTypes;
use native_gpt::{BurnModelConfig, TrainingConfig, train};

type InnerBackend = Wgpu<f32, i32>;
type OptimizerBackend = Autodiff<InnerBackend>;

const FILENAME: &str = "data/The_Verdict.txt";
fn main() {
    let file = std::fs::read_to_string(FILENAME).expect("Failed to read file");
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Unable to initiatlize tokenizer");
    let _input_tokens = tokenizer.encode(&file);
    let device = <InnerBackend as BackendTypes>::Device::default();
    train::<OptimizerBackend>(
        "artifacts",
        &file,
        &tokenizer,
        TrainingConfig::new(BurnModelConfig::new(), AdamWConfig::new()),
        &device,
    );
}
