use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use llm_helpers::gpt_model::GptModel;

fn generate_text_simple(
    model: &GptModel<ComputeBackend>,
    idx: Tensor<ComputeBackend, 2, Int>,
    max_new_tokens: usize,
    context_size: u32,
) -> Tensor<ComputeBackend, 2, Int> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        //let idx_dims = idx.dims();
        let idx_cond = idx.clone().slice(s![.., (-(context_size as i32))..-1]);
        let logits = model.forward(idx_cond);
        //let dims = logits.dims();
        let logits = logits.slice(s![.., -1, ..]).squeeze();
        let probas = burn::tensor::activation::softmax(logits, 2);
        let idx_next = probas.argmax(2);
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }
    idx
}

type B = Wgpu<f32, i32>;
type ComputeBackend = Autodiff<B>;
fn main() {
    let device = <ComputeBackend as BackendTypes>::Device::default();
    ComputeBackend::seed(&device, 123);
    println!("Hello, world!");
}
