use std::{fs::File, io::BufReader};

use polars::prelude::*;
use polars_io::prelude::*;

fn main() {
    let path = "data/ch06/sms+spam+collection/SMSSpamCollection";
    let parse_options = CsvParseOptions::default().with_separator('\t' as u8);
    let data_frame = CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(path.into()))
        .expect("Can not get file into reader")
        .finish()
        .expect("Can not create dataset");
    println!("Successfully loaded dataset");
    println!("{data_frame}");
}
