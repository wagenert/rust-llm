use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../../data/decoder_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
