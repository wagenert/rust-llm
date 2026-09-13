include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
use burn::backend::wgpu::Wgpu;
use burn::module::Module;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
// use burn_store::ModuleSnapshot;

type B = Wgpu<f32, i32>;

fn main() {
    let device = <B as BackendTypes>::Device::default();
    let model: Model<B> = Model::new(&device);
}
