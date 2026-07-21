use std::{fs::File, io::Read};
use regex::Regex;

const FILE_PATH: &str = "../data/The_Verdict.txt";
fn main() {
    let mut file = File::open(FILE_PATH).expect("Failed to open file");
    let mut content = String::new();
    let _ = file.read_to_string(&mut content).expect("Failed to read file");
    println!("Total characters in file: {}", content.len());
    println!("File content:\n{}", content.chars().take(100).collect::<String>());
    let re = Regex::new(r#"([,.:;?_!"()']|--|\s+)"#).unwrap();
    let split_text: Vec<&str> = re.split(&content).map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    //println!("Split text: {:?}", split_text[..10].join(", "));
    println!("Total words in file: {}", split_text.len());
    println!("First 10 words: {:?}", split_text[..30].join(", "));
}
