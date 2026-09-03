use std::ops::Index;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct NativeGptItem {
    input_ids: Vec<u32>,
    target_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct NativeGptDataset {
    items: Vec<NativeGptItem>,
}

impl NativeGptDataset {
    pub fn train(&self) -> Self {
        let split_idx = self.items.len() * 8 / 10;
        let train_items = self.items[..split_idx].to_vec();
        Self { items: train_items }
    }

    pub fn test(&self) -> Self {
        let split_idx = self.items.len() * 8 / 10;
        let test_items = self.items[split_idx..].to_vec();
        Self { items: test_items }
    }

    pub fn new(txt: &str, tokenizer: &tiktoken::CoreBpe, max_length: usize, stride: usize) -> Self {
        let token_ids = tokenizer.encode(txt);
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();
        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = token_ids[i..i + max_length].to_vec();
            let target_chunk = token_ids[i + 1..i + 1 + max_length].to_vec();
            input_ids.push(input_chunk);
            target_ids.push(target_chunk);
        }
        let items = input_ids
            .into_iter()
            .zip(target_ids.into_iter())
            .map(|(input_ids, target_ids)| NativeGptItem { input_ids, target_ids })
            .collect();
        Self { items }
    }
}

impl Index<usize> for NativeGptDataset {
    type Output = NativeGptItem;
    fn index(&self, index: usize) -> &Self::Output {
        &self.items[index]
    }
}

impl Dataset<NativeGptItem> for NativeGptDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<NativeGptItem> {
        self.items.get(index).cloned()
    }
}

#[derive(Clone, Debug)]
pub struct NativeGptBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub target_ids: Tensor<B, 1, Int>,
}

#[derive(Config, Default, Debug)]
pub struct NativeGptDataBatcher {
    #[config(default = 768)]
    max_length: usize,
}

impl<B: Backend> Batcher<B, NativeGptItem, NativeGptBatch<B>> for NativeGptDataBatcher {
    fn batch(&self, items: Vec<NativeGptItem>, device: &B::Device) -> NativeGptBatch<B> {
        let input_ids: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| &item.input_ids)
            .map(|inputs| TensorData::from(&inputs[..]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, self.max_length]))
            .collect();
        let target_ids: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| &item.target_ids)
            .map(|targets| TensorData::from(&targets[..]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        let input_ids = Tensor::cat(input_ids, 0);
        let target_ids = Tensor::cat(target_ids, 0);
        NativeGptBatch { input_ids, target_ids }
    }
}
