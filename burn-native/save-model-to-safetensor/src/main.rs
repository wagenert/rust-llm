use burn::backend::wgpu::Wgpu;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::NamedMpkFileRecorder;
use burn::tensor::backend::BackendTypes;
use burn_store::SafetensorsStore;
use clap::Parser;
use gpt_helpers::{BurnModel, BurnModelConfig};
use save_model_to_safetensor::parser::Cli;

type B = Wgpu<f32, i32>;

fn main() {
    let cli = Cli::parse();
    println!("Input path: {}", cli.input_file);
    println!("Output path: {}", cli.output_file);
    let device = <B as BackendTypes>::Device::default();
    let model: BurnModel<B> = BurnModelConfig::new().init(&device);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(cli.input_file, &recorder, &device)
        .expect("Can not load state from file");
}
