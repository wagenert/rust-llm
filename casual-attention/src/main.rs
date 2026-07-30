use burn::{Tensor, backend::Flex, tensor::{backend::BackendTypes, Float}};
use casual_attention::casual_attention::CasualAttention;

type MyBackend = Flex<f32, i32>;

fn main() {
     let inputs = [
        [0.43, 0.15, 0.89], 
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64], 
        [0.22, 0.58, 0.33], 
        [0.77, 0.25, 0.10], 
        [0.05, 0.80, 0.55], 
    ];

    let device = <Flex::<f32, i32> as BackendTypes>::Device::default();
    let input_tensor = Tensor::<MyBackend, 2, Float>::from_data(inputs, &device);
    let d_in = input_tensor.shape()[1];
    let d_out = 2;


    let sa_v1 = CasualAttention::<MyBackend>::new(d_in, d_out, 0.5, false, &device);
    let outputs = sa_v1.forward(input_tensor.clone());
    println!("Outputs: {}", outputs);
}
