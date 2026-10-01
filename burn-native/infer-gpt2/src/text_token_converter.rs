use burn::prelude::*;
use burn::tensor::{DType, DataError};
use std::string::FromUtf8Error;
use tiktoken::CoreBpe;

#[derive(Debug)]
pub enum TokenDecodingError {
    StringConversionError(String),
    DataConversionError(String),
}

impl From<FromUtf8Error> for TokenDecodingError {
    fn from(value: FromUtf8Error) -> Self {
        let utf8_error = value.utf8_error();
        TokenDecodingError::StringConversionError(format!(
            "Can not decode byte sequence to utf8. Sequence valid up to {}",
            utf8_error.valid_up_to()
        ))
    }
}

impl From<DataError> for TokenDecodingError {
    fn from(value: DataError) -> Self {
        TokenDecodingError::DataConversionError(format!("{value}"))
    }
}

pub struct TextTokenConverter<'a> {
    tokenizer: &'a CoreBpe,
}

impl<'a> TextTokenConverter<'a> {
    pub fn new(tokenizer_encoding: &str) -> Self {
        let tokenizer = tiktoken::get_encoding(tokenizer_encoding).unwrap();
        Self { tokenizer }
    }

    pub fn text_to_token_ids<B: Backend>(&self, text: &str, device: &B::Device) -> Tensor<B, 2, Int> {
        let encoded = self.tokenizer.encode_with_special_tokens(text);
        let encoded_tensor = Tensor::<B, 1, Int>::from_data(encoded.as_slice(), device).unsqueeze_dim::<2>(0);
        encoded_tensor
    }

    pub fn token_ids_to_text<B: Backend>(&self, token_ids: Tensor<B, 2, Int>) -> Result<String, TokenDecodingError> {
        let text_data = token_ids
            .squeeze_dim::<1>(0)
            .to_data()
            .convert_dtype(DType::U32)
            .to_vec::<u32>()?;
        let text_bytes = self.tokenizer.decode(text_data.as_slice());
        let decoded_text = String::from_utf8(text_bytes)?;
        Ok(decoded_text)
    }
}
