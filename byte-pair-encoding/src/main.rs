mod gpt_dataset;
const FILE_PATH: &str = "../data/The_Verdict.txt";
fn main() {
    let file = std::fs::File::open(FILE_PATH).expect("Failed to open file");
    let text = std::io::read_to_string(file).expect("Failed to read file");
    let tokenizer = tiktoken::get_encoding("gpt2").expect("Failed to get encoding");

    let enc_text = tokenizer.encode(&text);
    println!("Encoded text length: {:?}", enc_text.len());
    let enc_sample = enc_text[..50].to_vec();
    println!("Encoded sample: {:?}", enc_sample);
    let context_size = 4;
    let x = enc_sample[..context_size].to_vec();
    println!("Context: {:?}", x);
    let y = enc_sample[1..context_size + 1].to_vec();
    println!("Next token: {:?}", y);

    for i in 1..context_size + 1 {
        let context = enc_sample[..i].to_vec();
        let desired = enc_sample[i];
        println!("Context: {:?}, Next token: {:?}", context, desired);
    }
}
