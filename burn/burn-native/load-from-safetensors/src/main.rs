use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn_store::ModuleSnapshot;
use burn_store::PyTorchToBurnAdapter;
use burn_store::SafetensorsStore;
use gpt_helpers::BurnModel;
use gpt_helpers::BurnModelConfig;
use gpt_helpers::TextTokenConverter;
use gpt_helpers::generate_text_simple;

const MODEL_PATH: &str = "data/model.safetensors";

type WgpuBackend = burn::backend::wgpu::Wgpu<f32, i32>;

fn load_my_model<B: Backend>(model_path: &str, device: &B::Device) -> BurnModel<B> {
    let config = BurnModelConfig::new().with_qkv_bias(true);
    let mut model = config.init::<B>(device);
    let store = SafetensorsStore::from_file(model_path).with_from_adapter(PyTorchToBurnAdapter);
    let mut remapped_store = store
        .with_key_remapping(r"wte", "token_embedding")
        .with_key_remapping(r"wpe", r"positional_embedding")
        .with_key_remapping(r"\.h\.(\d+)\.", r".transformer_block.\1.");
    //.with_key_remapping(r"ln_f", r"output_layer");
    match model.load_from(&mut remapped_store) {
        Ok(apply_result) => println!("Success: {apply_result}"),
        Err(e) => {
            println!("{e}");
            panic!("Error loading model");
        }
    }
    model
}

fn main() {
    let device = <WgpuBackend as BackendTypes>::Device::default();
    let model = load_my_model::<WgpuBackend>(MODEL_PATH, &device);

    let token_converter = TextTokenConverter::new("gpt2");
    let input_ids = token_converter.text_to_token_ids::<WgpuBackend>("Every step moves you", &device);
    let output = generate_text_simple(&model, input_ids, 50, 1024);
    let output_text = token_converter
        .token_ids_to_text::<WgpuBackend>(output)
        .expect("Can not convert to string.");
    println!("Generated text: {}", output_text);
}
