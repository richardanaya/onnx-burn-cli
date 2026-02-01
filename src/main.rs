use anyhow::Result;
use clap::Parser;
use clap::Subcommand;

pub mod cli;
pub mod input;
pub mod ops;
pub mod runtime;

#[derive(Parser)]
#[command(name = "nnx")]
#[command(version = "0.0.0")]
#[command(about = "Neural Network Execute - GPU-accelerated ONNX inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available GPU devices
    Devices,
    /// Show information about a model
    Info { model: String },
    /// Perform inference using a model
    Infer {
        model: String,
        /// Input image file path
        #[arg(short = 'i', long)]
        input: Option<String>,
        /// Path to a labels file (one class per line)
        #[arg(short = 'l', long)]
        labels: Option<String>,
        /// Number of top predictions to show
        #[arg(short = 't', long, default_value = "5")]
        top: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Devices => {
            cli::devices::list_devices_impl();
        }
        Commands::Info { model } => {
            cli::info::info_impl(&model)?;
        }
        Commands::Infer {
            model,
            input,
            labels,
            top,
        } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cli::infer::infer_impl(&model, input.as_deref(), labels.as_deref(), top))?;
        }
    }

    Ok(())
}
