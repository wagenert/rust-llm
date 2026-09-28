use burn::backend::Autodiff;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn::tensor::Distribution;
use llm_helpers::gpt_config::GptConfig124M;
use transformer_block::transformer_block::TransformerBlock;

type B = burn::backend::flex::Flex<f32, i32>;
type ComputeBackend = Autodiff<B>;

fn main() {
    let gpt_config = GptConfig124M {
        vocab_size: 50257,
        context_length: 1024,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };
    let device = <ComputeBackend as BackendTypes>::Device::default();
    ComputeBackend::seed(&device, 0);
    let input = Tensor::<ComputeBackend, 3>::random(
        [2, gpt_config.context_length, gpt_config.emb_dim],
        Distribution::Default,
        &device,
    );
    let transformer_block = TransformerBlock::<ComputeBackend>::new(gpt_config, device);
    let output = transformer_block.forward(input.clone());
    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
}
