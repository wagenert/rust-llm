use crate::Gpt2Model;
use burn::prelude::*;

pub fn generate_text_simple<B: Backend>(
    model: &Gpt2Model<B>,
    idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: i32,
) -> Tensor<B, 2, Int> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., (-(context_size as i32))..-1]);
        println!("idx_cond shape: {}", idx_cond);
        let logits: Tensor<B, 2> = model.forward(idx_cond).squeeze_dim(0);
        let logits = logits.slice(s![-1, ..]);
        let probas = burn::tensor::activation::softmax(logits, 1);
        let idx_next = probas.argmax(1);
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }
    idx
}
