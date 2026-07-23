use std::ops::Index;

struct GPTDataset {
    ids: Vec<(Vec<u32>, Vec<u32>)>,
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

        let ids = input_ids.into_iter().zip(target_ids.into_iter()).map(|(input, target)| (input, target)).collect::<Vec<_>>();
        GPTDataset { ids }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

impl Index<usize> for GPTDataset {
    type Output = (Vec<u32>, Vec<u32>);

    fn index(& self, index: usize) -> & Self::Output {
        if index > self.len() {
            panic!("Index {} out of bounds for dataset of length {}", index, self.len());
        }
        &self.ids[index]
    }
}