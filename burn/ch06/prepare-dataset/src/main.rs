use std::{fs::File, io::BufReader};

use polars::prelude::*;
use polars_io::prelude::*;

fn main() {
    let path = "data/ch06/sms+spam+collection/SMSSpamCollection";
    let parse_options = CsvParseOptions::default()
        .with_separator('\t' as u8)
        .with_quote_char(None);
    let mut data_frame = CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(path.into()))
        .expect("Can not get file into reader")
        .finish()
        .expect("Can not create dataset");
    data_frame
        .set_column_names(&["category", "content"])
        .expect("Can not assign column names");
    println!("Successfully loaded dataset");
    println!("{data_frame}");
    println!("Column names {:?}", data_frame.columns());
    let categories = data_frame
        .group_by(["category"])
        .expect("Can not find colums category")
        .select(["content"])
        .count()
        .expect("Can not count content columns");
    println!("Categories: {}", categories);
    let spam = data_frame
        .clone()
        .lazy()
        .filter(col("category").eq(lit("spam")))
        .collect()
        .expect("Can not collect spam");
    let spam_length = spam.height();
    println!("Spam length: {}", spam_length);
    let ham = data_frame
        .lazy()
        .filter(col("category").eq(lit("ham")))
        .collect()
        .expect("Can not collect ham")
        .sample_n_literal(spam_length, false, Some(true), Some(123))
        .expect("Can not sample ham");
    println!("Ham length: {}", ham.height());
}
