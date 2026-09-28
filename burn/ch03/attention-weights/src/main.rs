use attention_weights::self_attention::SelfAttention;
use burn::{Tensor, backend::Flex, tensor::{Distribution, Float, activation::softmax, backend::BackendTypes}};

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
    let x_2 = input_tensor.clone().slice(1);
    let d_in = input_tensor.shape()[1];
    let d_out = 2;

    let distribution = Distribution::Uniform(0.0, 1.0);
    let w_query = Tensor::<MyBackend, 2, Float>::random(&[d_in, d_out], distribution, &device);
    let w_key = Tensor::<MyBackend, 2, Float>::random(&[d_in, d_out], distribution, &device);
    let w_value = Tensor::<MyBackend, 2, Float>::random(&[d_in, d_out], distribution, &device);

    let query_2 = x_2.clone().matmul(w_query);
    let key_2 = x_2.clone().matmul(w_key.clone());
    let value_2 = x_2.clone().matmul(w_value.clone());
    println!("Query: {}", query_2);
    println!("Key: {}", key_2);
    println!("Value: {}", value_2);

    let keys = input_tensor.clone().matmul(w_key);
    let values = input_tensor.clone().matmul(w_value);
    println!("Keys shape: {}", keys.shape());
    println!("Values shape: {}", values.shape());

    let keys_2 = keys.clone().slice(1);
    let attn_score_22 = query_2.clone().matmul(keys_2.clone().transpose());
    println!("Attention Score: {}", attn_score_22);

    let attn_scores_2 = query_2.clone().matmul(keys.clone().transpose());
    println!("Attention Scores: {}", attn_scores_2);

    let d_k = *keys.shape().last().unwrap_or(&1);
    let attn_weights_2: Tensor<Flex, 2> = softmax(attn_scores_2.clone() / (d_k as f32).sqrt(), 1);
    println!("Attention Weights: {}", attn_weights_2);

    let context_vec_2 = attn_weights_2.clone().matmul(values.clone());
    println!("Context Vector: {}", context_vec_2);

    let sa_v1 = SelfAttention::<MyBackend>::new(d_in, d_out, false, &device);
    let outputs = sa_v1.forward(input_tensor.clone());
    println!("Outputs: {}", outputs);
}
