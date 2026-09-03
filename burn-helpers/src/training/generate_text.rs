use crate::llm_layers::GptModel;
use burn::prelude::*;

pub fn generate_text_simple<B: Backend>(
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
