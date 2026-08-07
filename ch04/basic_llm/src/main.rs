use basic_llm::gpt_dummy_model::{GptConfig124M, GptDummyModel};
use burn::backend::Flex;
use burn::tensor::backend::BackendTypes;

static GPT_CONFIG_124M: GptConfig124M = GptConfig124M {
    vocab_size: 50257,
    context_length: 1024,
    emb_dim: 768,
    n_heads: 12,
    n_layers: 12,
    drop_rate: 0.1,
    qkv_bias: false,
};

type MyBackend = Flex<f32, i32>;

fn main() {
    let device = <MyBackend as BackendTypes>::Device::default();
    let model = GptDummyModel::<MyBackend>::new(&GPT_CONFIG_124M, device);
}
