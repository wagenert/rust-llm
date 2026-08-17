use std::any::{Any, TypeId};

use burn::tensor::backend::Backend;
use burn::{module::ModuleVisitor, optim::GradientsParams};

pub struct GradientInspector<'a> {
    pub grad_params: &'a GradientsParams,
}

impl<B: Backend> ModuleVisitor<B> for GradientInspector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<burn::prelude::Tensor<B, D>>) {
        if param.device() != B::Device::default() {
            println!("Parameter {} is on a different device than the model.", param.id);
            return;
        }
        /*if self.grad_params.type_id() != TypeId::of::<&burn::module::Param<burn::prelude::Tensor<B, D>>>() {
            println!(
                "Type mismatch for parameter {:?}: expected {:?}, found {:?}",
                param.id,
                self.grad_params.type_id(),
                TypeId::of::<&burn::module::Param<burn::prelude::Tensor<B, D>>>()
            );
            return;
        }*/
        if let Some(grad) = self.grad_params.get::<B, D>(param.id) {
            println!("Gradient for {}: {:?}", param.id, grad);
        } else {
            println!("No gradient found for {}", param.id);
        }
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &burn::module::Param<burn::prelude::Tensor<B, D, burn::prelude::Int>>,
    ) {
        if self.grad_params.type_id()
            != TypeId::of::<&burn::module::Param<burn::prelude::Tensor<B, D, burn::prelude::Int>>>()
        {
            println!(
                "Type mismatch for parameter {:?}: expected {:?}, found {:?}",
                param.id,
                self.grad_params.type_id(),
                TypeId::of::<&burn::module::Param<burn::prelude::Tensor<B, D, burn::prelude::Int>>>()
            );
            return;
        }
        if let Some(grad) = self.grad_params.get::<B, D>(param.id) {
            println!("Gradient for {}: {:?}", param.id, grad);
        } else {
            println!("No gradient found for {}", param.id);
        }
    }

    fn visit_bool<const D: usize>(
        &mut self,
        param: &burn::module::Param<burn::prelude::Tensor<B, D, burn::prelude::Bool>>,
    ) {
        if self.grad_params.type_id()
            != TypeId::of::<&burn::module::Param<burn::prelude::Tensor<B, D, burn::prelude::Bool>>>()
        {
            println!(
                "Type mismatch for parameter {:?}: expected {:?}, found {:?}",
                param.id,
                self.grad_params.type_id(),
                TypeId::of::<&burn::module::Param<burn::prelude::Tensor<B, D, burn::prelude::Bool>>>()
            );
            return;
        }
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
    /*
        fn visit_float_with_path<const D: usize>(
            &mut self,
            path: &[String],
            id: burn::module::ParamId,
            tensor: &burn::prelude::Tensor<B, D>,
        ) {
        }

        fn visit_int_with_path<const D: usize>(
            &mut self,
            path: &[String],
            id: burn::module::ParamId,
            tensor: &burn::prelude::Tensor<B, D, burn::prelude::Int>,
        ) {
        }

        fn visit_bool_with_path<const D: usize>(
            &mut self,
            path: &[String],
            id: burn::module::ParamId,
            tensor: &burn::prelude::Tensor<B, D, burn::prelude::Bool>,
        ) {
        }
    */
}
