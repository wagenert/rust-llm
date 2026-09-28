use candle_core::Tensor;

use crate::Gpt2Model;


pub fn generate_text_simple(
    model: &Gpt2Model,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: u32,
) -> candle_core::Result<Tensor> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let idx_cond = idx.narrow([.., (-(context_size as i32))..-1]);
        let logits = model.forward(idx_cond);
        let logits = logits.slice([.., -1, ..]);
        let probas = logits.softmax(logits, 2);
        let idx_next = probas.argmax(2).squeeze_dim(2);
        idx = Tensor::cat(vec![idx, idx_next], 1)?;
    }
    Ok(idx)
}
