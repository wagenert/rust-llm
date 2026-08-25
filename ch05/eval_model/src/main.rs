use burn::backend::autodiff::Autodiff;
use burn::backend::flex::Flex;
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn_helpers::GptConfig;
use burn_helpers::GptModel;

use eval_model::TextTokenConverter;

type B = Flex<f32, i32>;
type OptimizerBackend = Autodiff<B>;

fn generate_text_simple(
    model: &GptModel<B>,
    idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: u32,
) -> Tensor<B, 2, Int> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., (-(context_size as i32))..-1]);
        let logits = model.forward(idx_cond);
        let logits = logits.slice(s![.., -1, ..]);
        let probas = burn::tensor::activation::softmax(logits, 2);
        let idx_next = probas.argmax(2).squeeze_dim(2);
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }
    idx
}

fn main() {
    let tokenizer = TextTokenConverter::new("gpt2");
    let device = <OptimizerBackend as BackendTypes>::Device::default();
    OptimizerBackend::seed(&device, 123);
    let config = GptConfig::new();
    let model = config.init::<OptimizerBackend>(&device);
    let eval_model = model.valid();
    let start_context = "Every effort moves you";
    let token_ids = generate_text_simple(
        &eval_model,
        tokenizer.text_to_token_ids(start_context, &device),
        10,
        config.context_length as u32,
    );

    match tokenizer.token_ids_to_text(token_ids) {
        Ok(text) => println!("{text}"),
        Err(error) => println!("Error: {:?}", error),
    }
}
