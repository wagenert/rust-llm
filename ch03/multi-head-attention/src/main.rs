use burn::{backend::flex::Flex, tensor::backend::Backend};
use burn::tensor::backend::BackendTypes;
use burn::Tensor;
use burn::tensor::Float;
use multi_head_attention::multi_head_attention::{MultiHeadAttention, MultiHeadAttentionWrapper};

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
    MyBackend::seed(&device, 123);
    let input_tensor = Tensor::<MyBackend, 2, Float>::from_data(inputs, &device);

    let batch = Tensor::stack(vec![input_tensor.clone(), input_tensor.clone()], 0);
    println!("Batch: {}", batch);

    let shape = batch.shape();
    let context_length = shape[1];
    let d_in = shape[2];
    let _batch_size = shape[0];
    let d_out = 2;
    // let mha = MultiHeadAttentionWrapper::<MyBackend>::new(d_in, d_out, context_length, 0.0, 2, false, device);
    // let context_vecs = mha.forward(input_tensor);
    // println!("Context Vectors: {}", context_vecs);
    let mha = MultiHeadAttention::<MyBackend>::new(d_in, d_out, context_length, 0.0, 2, false, device);
    let context_vec = mha.forward(batch);
    println!("Context Vector: {}", context_vec);
}

