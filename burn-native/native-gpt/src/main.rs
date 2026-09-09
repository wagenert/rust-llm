use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::optim::AdamWConfig;
use burn::tensor::backend::BackendTypes;
use clap::Parser;
use native_gpt::{BurnModelConfig, TrainingConfig, train};
use native_gpt::{Cli, infer};

type InnerBackend = Wgpu<f32, i32>;
type OptimizerBackend = Autodiff<InnerBackend>;

const FILENAME: &str = "data/The_Verdict.txt";
fn main() {
    let args = Cli::parse();
    let file = std::fs::read_to_string(FILENAME).expect("Failed to read file");
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Unable to initiatlize tokenizer");
    let _input_tokens = tokenizer.encode(&file);
    let device = <InnerBackend as BackendTypes>::Device::default();
    match args.command {
        native_gpt::Commands::Train { output_path } => {
            train::<OptimizerBackend>(
                &output_path,
                &file,
                &tokenizer,
                TrainingConfig::new(BurnModelConfig::new(), AdamWConfig::new()).with_batch_size(2),
                &device,
            );
        }
        native_gpt::Commands::Infer { model_path, input_text } => {
            let input_tokens = tokenizer.encode(&input_text);
            let model_output = infer::<InnerBackend>(&model_path, device, input_tokens);
            let predicted = tokenizer.decode(&model_output);

            println!("Predicted {}", String::from_utf8_lossy(&predicted));
        }
    }
}
