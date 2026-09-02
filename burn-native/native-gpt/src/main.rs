use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use native_gpt::BurnModel;

type InnerBackend = Wgpu<f32, i32>;
type OptimizerBackend = Autodiff<InnerBackend>;

const FILENAME: &str = "../../data/The_Verdict.txt";
fn main() {
    let file = std::fs::read_to_string(FILENAME).expect("Failed to read file");
    println!("{}", file);
    let device = <OptimizerBackend as BackendTypes>::Device::default();
    OptimizerBackend::seed(&device, 123);
}
