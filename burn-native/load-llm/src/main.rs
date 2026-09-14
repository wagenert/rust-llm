pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model/decoder_model.rs"));
}
use burn::backend::wgpu::Wgpu;
use burn::tensor::backend::BackendTypes;
use model::Model;
// use burn_store::ModuleSnapshot;

type B = Wgpu<f32, i32>;

fn main() {
    let device = <B as BackendTypes>::Device::default();
    let _model: Model<B> = Model::new(&device);
}
