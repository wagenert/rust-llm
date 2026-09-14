use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn_helpers::GptConfig;
use burn_helpers::GptModel;
use burn_helpers::TextTokenConverter;
use burn_helpers::generate_text_simple;
use burn_store::ModuleSnapshot;
use burn_store::ModuleStore;
use burn_store::SafetensorsStore;

const MODEL_PATH: &str = "data/model.safetensors";

type WgpuBackend = burn::backend::wgpu::Wgpu<f32, i32>;

fn load_my_model<B: Backend>(model_path: &str, device: &B::Device) -> GptModel<B> {
    let config = GptConfig::new().with_qkv_bias(true);
    let mut model = GptModel::<B>::new(&config, device);
    let mut store = SafetensorsStore::from_file(model_path).allow_partial(true);
    /*let metadata = store.get_all_snapshots().expect("Unable to read snapshots from store.");
    println!("Metadata from the store: {:?}", metadata.keys());
    if let Ok(keys) = store.keys() {
        println!("Keys in the store: {:?}", keys);
    } else {
        println!("Failed to retrieve keys from the store.");
    }
    */
    let apply_result = model
        .load_from(&mut store)
        .expect("Failed to load model from safetensors");
    model
}

fn main() {
    let model = load_my_model::<WgpuBackend>(MODEL_PATH, &<WgpuBackend as BackendTypes>::Device::default());
    let device = <WgpuBackend as BackendTypes>::Device::default();
    let token_converter = TextTokenConverter::new("gpt2");
    let input_ids = token_converter.text_to_token_ids::<WgpuBackend>("Every step moves you", &device);
    let output = generate_text_simple(&model, input_ids, 50, 1024);
    let output_text = token_converter
        .token_ids_to_text::<WgpuBackend>(output)
        .expect("Can not convert to string.");
    println!("Generated text: {}", output_text);
}
