use basic_llm::example_deep_neural_network::ExampleDeepNeuralNetwork;
use basic_llm::gradient_inspector::GradientInspector;
use burn::backend::Flex;
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use llm_helpers::gpt_config::GptConfig124M;

static GPT_CONFIG_124M: GptConfig124M = GptConfig124M {
    vocab_size: 50257,
    context_length: 1024,
    emb_dim: 768,
    n_heads: 12,
    n_layers: 12,
    drop_rate: 0.1,
    qkv_bias: false,
};

type B = Flex<f32, i32>;
type ComputeBackend = burn::backend::Autodiff<B>;

fn print_gradients(
    model: ExampleDeepNeuralNetwork<ComputeBackend>,
    x: Tensor<ComputeBackend, 2>,
    device: &<ComputeBackend as BackendTypes>::Device,
) {
    let output = model.forward(x.clone());
    let target = Tensor::<ComputeBackend, 2>::from_data([[0.0]], device);

    let model_device = model.get_device();
    let outputs = output.to_device(&model_device);
    let targets = target.to_device(&model_device);

    let loss_fn = nn::loss::MseLoss::new();
    let loss = loss_fn.forward(outputs.clone(), targets, nn::loss::Reduction::Mean);
    let loss_value: f32 = loss.clone().mean_dim(0).to_data().as_slice().unwrap()[0];
    println!("Loss value: {}", loss_value);
    let grad = loss.backward();

    let loss_device = loss.device();

    if model_device != loss_device {
        panic!(
            "Model device ({:?}) and loss device ({:?}) do not match.",
            model_device, loss_device
        );
    }

    if let Some(grads) = outputs.grad(&grad) {
        println!("Gradients of the output: {:?}", grads);
    } else {
        println!("No gradients available for the output.");
    }

    let grad_params = GradientsParams::from_grads(grad, &model);
    let grad_params = grad_params.to_device(&model_device, &model);
    println!("Gradients of the model parameters: {:?}", grad_params);
    let mut inspector = GradientInspector {
        grad_params: &grad_params,
    };
    let device_model = model.to_device(device);
    device_model.visit(&mut inspector);
}

fn main() {
    let device = <ComputeBackend as BackendTypes>::Device::default();
    // let model = GptDummyModel::<MyBackend>::new(&GPT_CONFIG_124M, device);
    let layer_sizes = [3, 3, 3, 3, 3, 1];
    let sample_input = Tensor::<ComputeBackend, 2>::from_data([[1.0, 0.0, -1.0]], &device);
    ComputeBackend::seed(&device, 123);
    let model_without_shortcut =
        basic_llm::example_deep_neural_network::ExampleDeepNeuralNetwork::<ComputeBackend>::new(
            &layer_sizes,
            false,
            &device,
        );
    println!("Model without shortcut: {:?}", model_without_shortcut);
    print_gradients(model_without_shortcut, sample_input, &device);
}
