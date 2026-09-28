use burn::{module::Param, optim::GradientsParams, prelude::*};

pub struct GradientInspector<'a> {
    pub grad_params: &'a GradientsParams,
}

impl<B: Backend> burn::module::ModuleVisitor<B> for GradientInspector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        println!("Visiting parameter: {:?}", param);
        if let Some(grad) = self.grad_params.get::<B, D>(param.id) {
            println!("Gradient for {}: {:?}", param.id, grad);
        } else {
            println!("No gradient found for {}", param.id);
        }
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        println!("Visiting parameter: {:?}", param);
        if let Some(grad) = self.grad_params.get::<B, D>(param.id) {
            println!("Gradient for {}: {:?}", param.id, grad);
        } else {
            println!("No gradient found for {}", param.id);
        }
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        println!("Visiting parameter: {:?}", param);
        if let Some(grad) = self.grad_params.get::<B, D>(param.id) {
            println!("Gradient for {}: {:?}", param.id, grad);
        } else {
            println!("No gradient found for {}", param.id);
        }
    }

    fn enter_module(&mut self, name: &str, container_type: &str) {
        println!("Entering module: {} of type {}", name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        println!("Exiting module: {} of type {}", name, container_type);
    }
}
