use burn::prelude::*;
use full_model::gpt_model::GptModel;
use llm_helpers::gpt_config::GptConfig124M;

type B = burn::backend::Flex<f32, i32>;
type ComputeBackend = burn::backend::Autodiff<B>;
fn main() {
    let config = GptConfig124M {
        vocab_size: 50257,
        context_length: 1024,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };
    let device = <ComputeBackend as burn::tensor::backend::BackendTypes>::Device::default();
    ComputeBackend::seed(&device, 123);

    let model = GptModel::<ComputeBackend>::new(&config, device.clone());
}
