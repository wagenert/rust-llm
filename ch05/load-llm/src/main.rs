use burn::store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn::{
    backend::{Autodiff, Wgpu},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::BackendTypes,
};
use burn_helpers::{GptConfig, GptConfig124M, GptModel};

const MODEL_FILE: &str = "data/gpt2-small.safetensors";

type InnerBackend = Wgpu<f32, i32>;
type OptimizerBackend = Autodiff<Wgpu<f32, i32>>;

fn main() {
    let device = <OptimizerBackend as BackendTypes>::Device::default();
    let mut model = GptConfig::new().init::<OptimizerBackend>(&device);
    let mut store = SafetensorsStore::from_file(MODEL_FILE).with_from_adapter(PyTorchToBurnAdapter);
    model.load_from(&mut store).unwrap();
}
