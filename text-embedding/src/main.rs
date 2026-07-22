mod simple_tokenizers;
use std::{fs::File, io::Read};
use regex::Regex;

const FILE_PATH: &str = "../data/The_Verdict.txt";
fn main() {
    let mut file = File::open(FILE_PATH).expect("Failed to open file");
    let mut content = String::new();
    let _ = file.read_to_string(&mut content).expect("Failed to read file");
    println!("Total characters in file: {}", content.len());
    println!("File content:\n{}", content.chars().take(100).collect::<String>());
    let re = Regex::new(r#"([,.:;?!"()']|--|-|[A-Za-z0-9]+)"#).unwrap();
    let split_text: Vec<&str> = re.find_iter(&content).map(|s| s.as_str().trim()).filter(|s| !s.is_empty()).collect();
    println!("Total words in file: {}", split_text.len());
    println!("First 10 words: {:?}", split_text[..30].join(", "));
    let mut vocabulary: Vec<&str> = split_text.clone();
    vocabulary.sort();
    vocabulary.dedup();
    //println!("Unique words: {}", vocabulary.len());
    vocabulary.push("<|unk|>");
    vocabulary.push("<|endoftext|>"); // Add unknown token
    // println!("First 10 unique words: {:?}", vocabulary[..50].join(", "));
    let vocabulary_mapper: std::collections::HashMap<&str, usize> = vocabulary.into_iter().zip(0usize..).collect(); 
    //println!("Vocabulary mapper: {:?}", vocabulary_mapper);

    let tokenizer = simple_tokenizers::SimpleTokenizer::new(vocabulary_mapper);
    let ids = tokenizer.encode(&content);
    println!("Encoded IDs: {:?}", &ids[..30]);

    println!("Decoded text: {}", tokenizer.decode(&ids[..30]));

    let tokenizer = tiktoken::get_encoding("gpt2").unwrap();
    let text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace";
    let integers = tokenizer.encode_with_special_tokens(text);
    println!("Encoded integers: {:?}", integers);
    let decoded_text = tokenizer.decode_to_string(&integers).unwrap();
    println!("Decoded text: {}", decoded_text);
}
