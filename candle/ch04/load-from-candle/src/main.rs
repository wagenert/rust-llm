use candle_core::Device;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    // 1. Choose your execution device (CPU, CUDA, or Metal)
    let _device = Device::Cpu;

    // 2. Initialize the Hugging Face Hub API
    let hf_client = hf_hub::HFClient::new()?;

    // Choose a repository (using a tiny random BERT model for this example)
    let repo = hf_client.model("openai-community", "gpt2");
    let info = repo.info().send().await?;
    println!("{info:?}");
    // 3. Download the weights and the configuration file
    // These are saved to ~/.cache/huggingface/hub/ by default
    let weights_path = repo.get("model.safetensors").await?;
    let config_path = repo.get("config.json").await?;

        println!("Weights cached at: {:?}", weights_path);

        // 4. Read the configuration file (optional, but needed to build the model structure)
        let config_str = fs::read_to_string(config_path)?;
        println!("Config content: {}", config_str);

        // 5. Use VarBuilder to safely load the SafeTensors weights into the device
        // This handles mapping the file structures into Candle tensors automatically
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_path, candle_core::DType::F32, &device)? };

        println!("Successfully initialized VarBuilder with Hugging Face weights!");

        // You can now pass `vb` and your parsed config into your model structure!
        // Example: let model = MyModel::new(&config, vb)?;
    */
    Ok(())
}
