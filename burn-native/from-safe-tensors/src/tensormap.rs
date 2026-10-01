use burn::prelude::*;
use std::collections::HashMap;
use anyhow::{Context, Result};
use gpt_helpers::Gpt2Model;
use burn::{module::{Module, Param}, tensor::backend::Backend};

type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;

pub fn load_tensor_map(path: impl AsRef<std::path::Path>) -> Result<TensorMap> {
    let file = std::fs::File::open(path.as_ref()).with_context(|| format!("Cannot open {:?}", path.as_ref()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("Failed to map file {:?} to memory", path.as_ref()))?;
    let tensors = safetensors::SafeTensors::deserialize(&mmap).context("Failed to deserialize safen tensors.")?;
    let mut map = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let data = tensor.data();
        let floats = match dtype {
                safetensors::Dtype::F32 => {
                    data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>()
                },
                _ => anyhow::bail!("Unsupported tensor dtype: {:?}", dtype),
        };
        map.insert(name.to_string(), (shape, floats));
    }
    Ok(map)
}

pub fn get_tensor<B: Backend, const D: usize>(tensor_map: &TensorMap, key: &str, device: &B::Device) -> Result<Tensor<B, D>> {
    let (shape, data) = tensor_map.get(key).with_context(|| format!("Missing tensor for key: {}", key))?;
    let shape_arr: [usize; D] = shape.clone().try_into().map_err(|_| anyhow::anyhow!("Shape for key {} does not match expected dimension {}", key, D))?;
    let tensor = Tensor::from_data(TensorData::new(data.clone(), shape_arr), device);
    Ok(tensor)
}

pub fn load_gpt2_weights<B: Backend>(model: Gpt2Model<B>, tensormap: &TensorMap, device: &B::Device) -> Result<Gpt2Model<B>> {
    let model_to_load = model.clone();
    let mut model_record = model.into_record();
    let prefix = if tensormap.contains_key("transformer.wte.weight") {
        "transformer."
    } else if tensormap.contains_key("wte.weight") {
        ""
    } else {
        anyhow::bail!("Cannot find any known prefix in the tensor map.");
    };

    model_record.wte.weight = Param::from_tensor(get_tensor::<B, 2>(tensormap, &format!("{}wte.weight", prefix), device)?);
    model_record.wpe.weight = Param::from_tensor(get_tensor::<B, 2>(tensormap, &format!("{}wpe.weight", prefix), device)?);

    model_record.ln_f.gamma = Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}ln_f.weight", prefix), device)?);
    model_record.ln_f.beta = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}ln_f.bias", prefix), device)?));

    let lm_head_weight = if tensormap.contains_key(&format!("{}lm_head.weight", prefix))     {
        let tensor = get_tensor::<B, 2>(tensormap, &format!("{}lm_head.weight", prefix), device)?;
        tensor.transpose()
    } else {
        let wte = get_tensor::<B, 2>(tensormap, &format!("{}wte.weight", prefix), device)?;
        wte.transpose()
    };
    model_record.lm_head.weight = Param::from_tensor(lm_head_weight);
    model_record.lm_head.bias = None;

    let n_layers = model_record.transformers.len();
    for i in 0..n_layers {
        let block_prefix = format!("{}h.{}", prefix, i);
        let block = &mut model_record.transformers[i];
        block.ln_1.gamma = Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.ln_1.weight", block_prefix), device)?);
        block.ln_1.beta = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.ln_1.bias", block_prefix), device)?));

        block.ln_2.gamma = Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.ln_2.weight", block_prefix), device)?);
        block.ln_2.beta = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.ln_2.bias", block_prefix), device)?));

        block.attn.c_attn.weight = Param::from_tensor(get_tensor::<B, 2>(tensormap, &format!("{}.attn.c_attn.weight", block_prefix), device)?);
        block.attn.c_attn.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.attn.c_attn.bias", block_prefix), device)?));
        block.attn.c_proj.weight = Param::from_tensor(get_tensor::<B, 2>(tensormap, &format!("{}.attn.c_proj.weight", block_prefix), device)?);
        block.attn.c_proj.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.attn.c_proj.bias", block_prefix), device)?));
        
        block.mlp.c_fc.weight = Param::from_tensor(get_tensor::<B, 2>(tensormap, &format!("{}.mlp.c_fc.weight", block_prefix), device)?);
        block.mlp.c_fc.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.mlp.c_fc.bias", block_prefix), device)?));
        block.mlp.c_proj.weight = Param::from_tensor(get_tensor::<B, 2>(tensormap, &format!("{}.mlp.c_proj.weight", block_prefix), device)?);
        block.mlp.c_proj.bias = Some(Param::from_tensor(get_tensor::<B, 1>(tensormap, &format!("{}.mlp.c_proj.bias", block_prefix), device)?));
    }
    Ok(model_to_load.load_record(model_record))
}