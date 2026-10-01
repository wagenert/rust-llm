use burn::prelude::*;
use burn::record::Recorder;
use burn::tensor::DType;
use burn::{record::CompactRecorder, tensor::backend::Backend};

use crate::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, input_tokens: Vec<u32>) -> Vec<u32> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);
    let input_tensor = Tensor::<B, 1, Int>::from_data(input_tokens.as_slice(), &device).unsqueeze();
    let output = model.forward(input_tensor);
    let predicted = output
        .argmax(1)
        .flatten::<1>(0, 1)
        .to_data()
        .convert_dtype(DType::U32)
        .to_vec();
    predicted.expect("Failed to convert tensor to data")
}
