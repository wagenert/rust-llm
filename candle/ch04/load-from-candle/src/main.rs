use candle_core::Device;
use candle_nn::VarBuilder;
use std::fs;
use load_from_candle::Gpt2Model;

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
    
    let config_str = fs::read_to_string(config_path)?;
    println!("Config content: {}", config_str);

    // 5. Use VarBuilder to safely load the SafeTensors weights into the device
    // This handles mapping the file structures into Candle tensors automatically
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device).expect("Can not initialize weights")
    };

    println!("Successfully initialized VarBuilder with Hugging Face weights!");

    // You can now pass `vb` and your parsed config into your model structure!
    // Example: let model = MyModel::new(&config, vb)?;
    let n_heads = 12;
    let n_embed = 768;
    let model = Gpt2Model::new(n_heads, n_embed, vb);
    println!("Successfully initialized model.");
    Ok(())
}
