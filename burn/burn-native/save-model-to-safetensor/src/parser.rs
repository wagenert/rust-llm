use clap::Parser;

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(help = "Input file")]
    pub input_file: String,
    #[arg(help = "Output file")]
    pub output_file: String,
}
