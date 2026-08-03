use burn::backend::flex::Flex;
use burn::tensor::backend::BackendTypes;
use burn::Tensor;
use burn::tensor::Float;
use multi_head_attention::multi_head_attention::MultiHeadAttentionWrapper;

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

    let context_length = 5;
    let d_in = 3;
    let d_out = 2;
    let mha = MultiHeadAttentionWrapper::<MyBackend>::new(d_in, d_out, context_length, 0.0, 2, false, device);
    let context_vecs = mha.forward(input_tensor);
    println!("Context Vectors: {:?}", context_vecs);
}
