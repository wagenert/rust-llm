use burn::module::Module;
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
struct ExampleNeuralNetworkLayer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Gelu,
}

impl<B: Backend> ExampleNeuralNetworkLayer<B> {
    pub fn new(layer_size1: usize, layer_size2: usize, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(layer_size1, layer_size2).init(device);
        let linear2 = LinearConfig::new(layer_size2, layer_size1).init(device);
        let activation = Gelu::new();
        Self {
            linear1,
            linear2,
            activation,
        }
    }

    fn forward(&self, input: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }

    pub fn get_layer_device<'a>(&self) -> B::Device {
        self.linear1.weight.device().clone()
    }
}

#[derive(Module, Debug)]
pub struct ExampleDeepNeuralNetwork<B: Backend> {
    use_shortcut: bool,
    layers: Vec<ExampleNeuralNetworkLayer<B>>,
}

impl<B: Backend> ExampleDeepNeuralNetwork<B> {
    pub fn new(layer_sizes: &[usize], use_shortcut: bool, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            layers.push(ExampleNeuralNetworkLayer::new(
                layer_sizes[i - 1],
                layer_sizes[i],
                device,
            ));
        }
        Self {
            use_shortcut: use_shortcut,
            layers,
        }
    }

    pub fn forward(&self, input: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        let mut x = input;
        for layer in &self.layers {
            let layer_output = layer.forward(x.clone());
            if self.use_shortcut {
                x = x + layer_output;
            } else {
                x = layer_output;
            }
        }
        x
    }

    pub fn get_model_device(&self) -> B::Device {
        self.layers[0].get_layer_device()
    }
}
