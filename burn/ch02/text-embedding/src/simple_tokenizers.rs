use std::collections::HashMap;

use regex::Regex;

pub struct SimpleTokenizer<'a> {
    vocabulary: HashMap<&'a str, usize>,
    reverse_vocabulary: HashMap<usize, &'a str>,
}

impl<'a> SimpleTokenizer<'a> {
    pub fn new(vocabulary: HashMap<&'a str, usize>) -> Self {
        let reverse_vocabulary = vocabulary.iter().map(|(k, &v)| (v, *k)).collect();
        Self { vocabulary, reverse_vocabulary }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let re = Regex::new(r#"([,.:;?!\"()\']|--|-|[\w\d]+)"#).unwrap();
        re.find_iter(text)
            .map(|s| s.as_str().trim())
            .filter(|s| !s.is_empty())
            .map(|word| *self.vocabulary.get(word).unwrap_or(self.vocabulary.get("<|unk|>").unwrap())) // Use 0 for unknown words
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .map(|&token| *self.reverse_vocabulary.get(&token).unwrap_or(&"<|unk|>"))
            .collect::<Vec<&str>>()
            .join(" ")
    }
}