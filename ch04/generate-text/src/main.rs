use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::backend::BackendTypes;
use llm_helpers::gpt_config::GptConfig124M;
use llm_helpers::gpt_model::GptModel;

fn generate_text_simple(
    model: &GptModel<ComputeBackend>,
    idx: Tensor<ComputeBackend, 2, Int>,
    max_new_tokens: usize,
    context_size: u32,
) -> Tensor<ComputeBackend, 2, Int> {
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

type B = Wgpu<f32, i32>;
type ComputeBackend = Autodiff<B>;
fn main() {
    let config = GptConfig124M {
        vocab_size: 50257,
        context_length: 1024,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };

    let tokenizer = tiktoken::get_encoding("gpt2").unwrap();
    let starting_context = "Hello, I am";
    let encoded_vector = tokenizer.encode(starting_context);
    println!("Encoded vector: {encoded_vector:?}");

    let device = <ComputeBackend as BackendTypes>::Device::default();
    let encoded_tensor =
        Tensor::<ComputeBackend, 1, Int>::from_data(encoded_vector.as_slice(), &device).unsqueeze::<2>();
    println!("Encoded tensor shape: {:?}", encoded_tensor.shape());
    ComputeBackend::seed(&device, 123);
    let model = GptModel::<ComputeBackend>::new(&config, device.clone());
    let out = generate_text_simple(&model, encoded_tensor, 6, config.context_length as u32);
    let squeezed_tensor = out.squeeze_dim::<1>(0);
    let out_vector = squeezed_tensor
        .to_data()
        .convert_dtype(DType::U32)
        .to_vec::<u32>()
        .expect("Can not convert to i32");

    println!("Decoded vector {:?}", out_vector);

    let decoded_tokens = tokenizer.decode(&out_vector);
    let decoded_text = String::from_utf8(decoded_tokens).unwrap();
    println!("Decoded text: {}", decoded_text);
}
