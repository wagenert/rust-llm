use burn::backends::flex::Flex;
use burn::prelude::*;
use llm_helpers::gpt_config::GPTConfig124M;

type B = Flex<f32, i32>;
type ComputeBackend = Autodiff<B>;
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
    let device = <ComputeBackend as BackendTypes>::Device::default();
    ComputeBackend::seed(device, 123);
}
