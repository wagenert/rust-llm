use std::string::FromUtf8Error;
use candle_core::{Device, Tensor};
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

/*impl From<DataError> for TokenDecodingError {
    fn from(value: DataError) -> Self {
        TokenDecodingError::DataConversionError(format!("{value}"))
    }
}*/

pub struct TextTokenConverter<'a> {
    tokenizer: &'a CoreBpe,
}

impl<'a> TextTokenConverter<'a> {
    pub fn new(tokenizer_encoding: &str) -> Self {
        let tokenizer = tiktoken::get_encoding(tokenizer_encoding).unwrap();
        Self { tokenizer }
    }

    pub fn text_to_token_ids(&self, text: &str) -> candle_core::Result<Tensor> {
        let encoded = self.tokenizer.encode_with_special_tokens(text);
        let encoded_tensor = Tensor::from_vec(encoded, 1, &Device::Cpu)?;
        Ok(encoded_tensor)
    }

    pub fn token_ids_to_text(&self, token_ids: Tensor) -> candle_core::Result<String> {
        let text_data = token_ids.to_vec1::<u32>()?;
        let text_bytes = self.tokenizer.decode(text_data.as_slice());
        let decoded_text = String::from_utf8(text_bytes)?;
        Ok(decoded_text)
    }
}
