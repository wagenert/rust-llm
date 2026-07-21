use std::collections::HashMap;

use regex::Regex;

pub struct SimpleTokenizerV1<'a> {
    vocabulary: HashMap<&'a str, usize>,
    reverse_vocabulary: HashMap<usize, &'a str>,
}

impl<'a> SimpleTokenizerV1<'a> {
    pub fn new(vocabulary: HashMap<&'a str, usize>) -> Self {
        let reverse_vocabulary = vocabulary.iter().map(|(k, &v)| (v, *k)).collect();
        SimpleTokenizerV1 { vocabulary, reverse_vocabulary }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let re = Regex::new(r#"([,.:;?!\"()\']|--|-|[\w\d]+)"#).unwrap();
        re.find_iter(text)
            .map(|s| s.as_str().trim())
            .filter(|s| !s.is_empty())
            .map(|word| *self.vocabulary.get(word).unwrap_or(&0)) // Use 0 for unknown words
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .map(|&token| *self.reverse_vocabulary.get(&token).unwrap_or(&"<UNK>"))
            .collect::<Vec<&str>>()
            .join(" ")
    }
}