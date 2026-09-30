use candle_core::Device;
use candle_nn::VarBuilder;
use load_from_candle::{Gpt2Config, Gpt2Model, TextTokenConverter};
use std::fs::File;
use std::io::BufReader;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    // 1. Choose your execution device (CPU, CUDA, or Metal)
    let device = Device::Cpu;

    // 2. Initialize the Hugging Face Hub API
    let hf_client = hf_hub::HFClient::new()?;

    // Choose a repository (using a tiny random BERT model for this example)
    let repo = hf_client.model("openai-community", "gpt2");
    // let info = repo.info().send().await?;
    // println!("{info:?}");
    // 3. Download the weights and the configuration file
    // These are saved to ~/.cache/huggingface/hub/ by default
    let weights_path = repo.download_file().filename("model.safetensors").send().await?;

    let config_path = repo.download_file().filename("config.json").send().await?;

    // 4. Read the configuration file (optional, but needed to build the model structure)
    println!("Weights cached at: {:?}", weights_path);

    let file = File::open(config_path)?;
    let config_reader = BufReader::new(file);
    let config: Gpt2Config = serde_json::from_reader(config_reader)?;
    println!("Read config from file: {config:?}");

    // 5. Use VarBuilder to safely load the SafeTensors weights into the device
    // This handles mapping the file structures into Candle tensors automatically
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
            .expect("Can not initialize weights")
    };

    println!("Successfully initialized VarBuilder with Hugging Face weights!");

    // You can now pass `vb` and your parsed config into your model structure!
    // Example: let model = MyModel::new(&config, vb)?;
    match Gpt2Model::new(&config, vb) {
        Ok(model) => {
            println!("Successfully initialized model.");
            let input_text = "Every step moves you";
            let text_token_converter = TextTokenConverter::new("gpt2");
            match text_token_converter.text_to_token_ids(input_text) {
                Ok(input_ids) => {
                    println!("Successfully encoded text {input_ids}");
                    match model.forward(&input_ids) {
                        Ok(output_ids) =>  {
                            println!("Successfully created output ids {output_ids:?}");
                            if let Ok(output_text) = text_token_converter.token_ids_to_text(output_ids) {
                                println!("Input {input_text}");
                                println!("Output {output_text}");
                            } else {
                                panic!("Can not decode output ids");
                            }
                        },
                        Err(e) => {
                            println!("Can not generate text. Error {e}");
                        }
                    }
                },
                Err(e) => println!("Can not encode input text. Error {e}")
            }
        }
        Err(e) => println!("Can not create model. Error {e}"),
    }

    Ok(())
}
