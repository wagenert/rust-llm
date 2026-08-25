use burn::backend::flex::Flex;
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use std::fs;

type B = Flex<f32, i32>;
type OptimizeBackend = Autodiff<B>;

const FILEPATH: &str = "../../../data/The_Verdict.txt";

fn main() {
    let text = fs::read_to_string(FILEPATH);
    let device = <OptimizeBackend as BackendTypes>::Device::default();
    OptimizeBackend::seed(&device, 123);

    println!("Hello, world!");
}
