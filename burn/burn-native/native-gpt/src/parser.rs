use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    Train { output_path: String },
    Infer { model_path: String, input_text: String },
}
