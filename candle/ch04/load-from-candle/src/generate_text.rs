use candle_core::Tensor;
use candle_nn::ops::softmax;

use crate::Gpt2Model;


pub fn generate_text_simple(
    model: &Gpt2Model,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: u32,
) -> candle_core::Result<Tensor> {
    println!("Received tensor {idx}");
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let idx_cond = idx.narrow(1, idx.dim(1)? - context_size as usize, idx.dim(1)?)?;
        //let idx_cond = idx.slice([.., (-(context_size as i32))..-1]);
        let logits = model.forward(&idx_cond)?;
        //let logits = logits.slice([.., -1, ..]);
        let logits = logits.narrow(1, logits.dim(1)? - 1, logits.dim(1)?)?;
        let probas = softmax(&logits,2)?;
        let idx_next = probas.argmax(2)?.squeeze(2)?;
        idx = Tensor::cat(&[&idx, &idx_next], 1)?;
    }
    Ok(idx)
}
