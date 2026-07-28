use burn::{Tensor, backend::Flex, tensor::{Float, backend::BackendTypes}};

type MyBackend = Flex<f32, i32>;

fn main() {
    let device = <MyBackend as BackendTypes>::Device::default();
    let t1_vec = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let t2_vec = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let t1 = Tensor::<MyBackend, 2, Float>::from_data(t1_vec, &device);
    let t2 = Tensor::<MyBackend, 2, Float>::from_data(t2_vec, &device);
    println!("t1: {}", t1);
    println!("t2: {}", t2);

    let t3 = t1.clone() * t2.clone();
    println!("t3: {}", t3);

    let t4 = t1.clone().matmul(t2.clone().transpose());
    println!("t4: {}", t4);
}
