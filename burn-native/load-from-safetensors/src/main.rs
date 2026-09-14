use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn_helpers::GptConfig;
use burn_helpers::GptModel;
use burn_store::ModuleSnapshot;
use burn_store::ModuleStore;
use burn_store::SafetensorsStore;

const MODEL_PATH: &str = "../../../data/model.safetensors";

type WgpuBackend = burn::backend::wgpu::Wgpu<f32, i32>;

fn load_my_model<B: Backend>(model_path: &str, device: &B::Device) -> GptModel<B> {
    let config = GptConfig::new();
    let mut model = GptModel::<B>::new(&config, device);
    let mut store = SafetensorsStore::from_file(model_path).allow_partial(true);
    for key in store.keys() {
        println!("Key: {:?}", key);
    }
    let apply_result = model
        .load_from(&mut store)
        .expect("Failed to load model from safetensors");
    model
}

fn main() {
    let model = load_my_model::<WgpuBackend>(MODEL_PATH, &<WgpuBackend as BackendTypes>::Device::default());
}
