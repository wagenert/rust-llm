use crate::Gpt2Model;
use burn::prelude::*;
use burn::tensor::{backend::Backend, Distribution, Tensor};

/// Samples one index from a 2D tensor of probabilities [batch_size, num_classes]
/// using the Gumbel-Max trick.
fn burn_multinomial_one_sample<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 2, burn::tensor::Int> {
    let device = probs.device();
    let shape = probs.shape();

    // 1. Generate Uniform(0, 1) noise matching the probability shape
    let uniform = Tensor::<B, 2>::random(shape, Distribution::Default, &device);

    // 2. Transform Uniform noise into Gumbel noise: -log(-log(U))
    let eps = 1e-20; // Prevent log(0)
    let gumbel = -(-uniform.clone().clamp(eps, 1.0).log()).clamp(eps, f64::MAX).log();

    // 3. Add Gumbel noise to log probabilities and take the argmax
    let logits = probs.log() + gumbel;
    
    // Returns a tensor of sampled indices with shape [batch_size, 1]
    logits.argmax(1)
}

pub fn generate_text_simple<B: Backend>(
    model: &Gpt2Model<B>,
    idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: i32,
    temperature: Option<f32>,
    top_k: Option<usize>,
    eos_id: Option<i32>
) -> Tensor<B, 2, Int> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., (-(context_size as i32))..-1]);
        let logits: Tensor<B, 2> = model.forward(idx_cond).squeeze_dim(0);
        let mut logits = logits.slice(s![-1, ..]);
        if let Some(top_k) = top_k {
            let top_k_tensor = logits.clone().topk(top_k, 1);
            let min_val = top_k_tensor.min().try_into_scalar().expect("Failed to convert min value to scalar");
            logits = logits.clone().mask_fill(logits.lower_elem(min_val), f32::NEG_INFINITY);
        }
        
        let idx_next = if let Some(temperature) = temperature {
            if temperature <= 0.0 {
                panic!("Temperature must be greater than 0");
            } else {
                logits = logits / temperature;
                let probas = burn::tensor::activation::softmax(logits, 1);
                burn_multinomial_one_sample(probas)
            }
        } else {
            let probas = burn::tensor::activation::softmax(logits, 1);
            probas.argmax(1)
        };


        if let Some(end_of_sequence_id) = eos_id {
            let idx_next_scalar = idx_next.clone().slice(s![0, 0]).try_into_scalar().expect("Failed to convert idx_next to scalar").to_i32();
            if idx_next_scalar == end_of_sequence_id {
                return idx;
            }
        }
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }
    idx
}
