use basic_llm::example_deep_neural_network::ExampleDeepNeuralNetwork;
use basic_llm::gpt_dummy_model::{GptConfig124M, GptDummyModel};
use burn::Tensor;
use burn::backend::Flex;
use burn::nn::loss::{MseLoss, Reduction};
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

fn print_gradients(model: ExampleDeepNeuralNetwork<MyBackend>, x: Tensor::<MyBackend, 2>, device: &<MyBackend as BackendTypes>::Device) {
    let output = model.forward(x.clone());
    let target = Tensor::<MyBackend, 2>::from_data([[0.0]], device);

    let mut loss = MseLoss::new();
    loss = loss.forward(output.clone(), target.clone(), Reduction::Auto);
    loss.backward();

    for name, param in model.named_parameters() {
        let grad = param.grad();
        println!("Gradient for {}: {:?}", name, grad);
    }

}
fn main() {
    let device = <MyBackend as BackendTypes>::Device::default();
    // let model = GptDummyModel::<MyBackend>::new(&GPT_CONFIG_124M, device);
    let layer_sizes = [3, 3, 3, 3, 3, 1];
    //MyBackend::seed(123);
    let model_without_shortcut = basic_llm::example_deep_neural_network::ExampleDeepNeuralNetwork::<MyBackend>::new(&layer_sizes, false, &device);
    println!("{:?}", model_without_shortcut);
}
