use burn::Tensor;
// use burn::backend::Wgpu;
// use burn::backend::wgpu::WgpuDevice;
use burn::backend::flex::Flex;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{ElementConversion, Float, Int};

//type MyBackend = Wgpu<f32, i32>;
type MyBackend = Flex<f32, i32>;
// const FILE_PATH: &str = "../data/The_Verdict.txt";

fn main() {
    // let file = std::fs::File::open(FILE_PATH).expect("Failed to open file");
    // let txt = std::io::read_to_string(file).expect("Failed to read file");

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
    let query = input_tensor.clone().slice(1);
    println!("Input tensor: {}", &input_tensor);
    println!("Query: {}", query);

    let mut scores_vec = Vec::with_capacity(input_tensor.dims()[0]);
    for x_i in input_tensor.clone().iter_dim(0) {
        let score = x_i.matmul(query.clone().transpose()).sum();
        scores_vec.push(score);
    }
    let attn_scores_2 = Tensor::cat(scores_vec, 0);
    println!("Attention scores: {}", &attn_scores_2);

    /* 
    let attn_scores_2_sum = attn_scores_2.clone().sum().try_into_scalar().expect("Failed to convert to scalar");
    let attn_weights_2 = attn_scores_2.clone().div_scalar(attn_scores_2_sum);
    println!("Attention weights: {}", &attn_weights_2);
    */

    let attn_weights_2: Tensor<MyBackend, 1, Float> = burn::tensor::activation::softmax(attn_scores_2.clone(), 0);
    // println!("Attention scores (softmax): {}", &attn_weights_2);
    let attn_weights_reshaped = attn_weights_2.clone().reshape([attn_weights_2.dims()[0], 1]);
    println!("Attention weights (reshaped): {}", &attn_weights_reshaped.clone().transpose());

    //let mut context_vec_2 = Vec::new();
    for (i, x_i) in input_tensor.clone().iter_dim(1).enumerate() {
        println!("Index: {}, x_i: {}", i, &x_i);
        //let indices = Tensor::<MyBackend, 1, Int>::from_data([i as i32], &device);
        //let attn_weight: Tensor<MyBackend, 1, Float> = attn_weights_2.clone().select(0, indices);
        let weighted_x_i = x_i.matmul(attn_weights_reshaped.clone().transpose());
        println!("Weighted x_i: {}", &weighted_x_i);
        //context_vec_2.push(weighted_x_i);
    }
    //let context_tensor: Tensor<MyBackend, 1, Float> = Tensor::cat(context_vec_2, 0);
    //println!("Context vector: {}", &context_tensor);
}
