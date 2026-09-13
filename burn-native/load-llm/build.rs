use burn_onnx::ModelGen;
use std::env;

fn main() {
    let key = "OUT_DIR";
    if env::var(key).is_err() {
        unsafe {
            env::set_var(key, "burn-native/load-llm/src");
        }
    }

    ModelGen::new()
        .input("../../data/model.onnx")
        .out_dir("model/")
        .run_from_script();
}
