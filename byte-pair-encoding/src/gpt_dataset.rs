use std::ops::Index;
use std::sync::Arc;

use burn::Tensor;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::tensor::{Int, TensorData};
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct GPTItem {
    input_ids: Vec<u32>,
    target_ids: Vec<u32>,
}
pub struct GPTDataset {
    items: Vec<GPTItem>,
}

impl GPTDataset {
    fn create(txt: &str, tokenizer: &tiktoken::CoreBpe, max_length: usize, stride: usize) -> Self {
        let token_ids = tokenizer.encode(txt);
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();

        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = token_ids[i..i + max_length].to_vec();
            let target_chunk = token_ids[i + 1..i + 1 + max_length].to_vec();
            input_ids.push(input_chunk);
            target_ids.push(target_chunk);
        }

        let items = input_ids.into_iter().zip(target_ids.into_iter()).map(|(input, target)| GPTItem { input_ids: input, target_ids: target }).collect::<Vec<_>>();
        GPTDataset { items }
    }

}

impl Index<usize> for GPTDataset {
    type Output = GPTItem;

    fn index(& self, index: usize) -> & Self::Output {
        if index >= self.items.len() {
            panic!("Index {} out of bounds for dataset of length {}", index, self.items.len());
        }
        &self.items[index]
    }
}

impl Dataset<GPTItem> for GPTDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<GPTItem> 
    {
        if index < self.len() {
            Some(self.items[index].clone())
        } else {
            None
        }
    }

}

#[derive(Clone, Debug)]
pub struct GPTBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub target_ids: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
pub struct GPTBatcher {
    max_length: usize,
}

impl GPTBatcher {
    fn new(max_length: usize) -> Self {
        GPTBatcher { max_length }
    }
}

impl<B: Backend> Batcher<B, GPTItem, GPTBatch<B>> for GPTBatcher {
    fn batch(&self, items: Vec<GPTItem>, device: &B::Device) -> GPTBatch<B> {
        let input_ids: Vec<Tensor<B, 2, Int>> = items.iter().map(|item| &item.input_ids)
            .map(|inputs| TensorData::from(&inputs[..]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, self.max_length]))
            .collect();
        let target_ids: Vec<Tensor<B, 2, Int>> = items.iter().map(|item| &item.target_ids)
            .map(|targets| TensorData::from(&targets[..]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, self.max_length]))
            .collect();

        let input_ids = Tensor::cat(input_ids, 0);
        let target_ids = Tensor::cat(target_ids, 0);
        GPTBatch { input_ids, target_ids }
    }
}

pub fn create_dataloader<B: Backend>(txt: &str, batch_size: usize, max_length: usize, stride: usize, shuffle: bool, num_workers: usize, device: B::Device) -> Arc<dyn DataLoader<B, GPTBatch<B>>> {
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Failed to get encoding");
    let dataset = Arc::new(GPTDataset::create(txt, &tokenizer, max_length, stride));
    // let strategy = Box::new(FixBatchStrategy::<u32>::new(batch_size));

    let batcher = GPTBatcher::new(max_length);
    let dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = 
        if shuffle {
            DataLoaderBuilder::new(batcher)
                .batch_size(batch_size)
                .num_workers(num_workers)
                .set_device(device)
                .shuffle(123)
                .build(dataset)
        } else {
            DataLoaderBuilder::new(batcher)
                .batch_size(batch_size)
                .num_workers(num_workers)
                .set_device(device)
                .build(dataset)
        };

    dataloader
}