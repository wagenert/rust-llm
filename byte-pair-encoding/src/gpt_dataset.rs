use std::ops::Index;

struct GPTDataset {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
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

        GPTDataset { input_ids, target_ids }
    }

    fn len(&self) -> usize {
        self.input_ids.len()
    }
}

impl<'a> Index<usize> for GPTDataset {
    type Output = (&'a Vec<u32>, &'a Vec<u32>);

    fn index(&self, index: usize) -> &Self::Output {
        if index > self.len() {
            panic!("Index {} out of bounds for dataset of length {}", index, self.len());
        }
        &(&self.input_ids[index], &self.target_ids[index])
    }
}