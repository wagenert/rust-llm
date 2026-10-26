use std::fs::File;

use polars::prelude::*;

struct BalancedDataSet {
    dfs: Vec<DataFrame>,
}

impl BalancedDataSet {
    pub fn create(df: DataFrame, column: &str) -> PolarsResult<Self> {
        let dfs = df.partition_by([column], true)?;
        Ok(Self { dfs })
    }

    fn get_balanced_sample(&self) -> Result<Vec<DataFrame>, PolarsError> {
        if let Some(min_height) = self.dfs.iter().map(|df| df.height()).min() {
            let mut balanced_dataset = Vec::with_capacity(self.dfs.len());
            for df in self.dfs.clone() {
                if df.height() == min_height {
                    balanced_dataset.push(df);
                    continue;
                }
                let pruned_df = df.sample_n_literal(min_height, false, Some(true), Some(123))?;
                balanced_dataset.push(pruned_df);
            }
            Ok(balanced_dataset)
        } else {
            Err(PolarsError::NoData("No data in dataset".into()))
        }
    }
}

fn load_csv_dataframe_from_file(path: &str, read_options: CsvReadOptions) -> Result<DataFrame, PolarsError> {
    read_options.try_into_reader_with_file_path(Some(path.into()))?.finish()
}

const COLUMN_NAME: &str = "category";
fn main() -> PolarsResult<()> {
    let path = "data/ch06/sms+spam+collection/SMSSpamCollection";
    let parse_options = CsvParseOptions::default()
        .with_separator('\t' as u8)
        .with_quote_char(None);
    let read_options = CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_options);
    let mut df = load_csv_dataframe_from_file(path, read_options)?;
    df.set_column_names(&["category", "content"])?;
    println!("Successfully loaded dataset");
    println!("{df}");
    println!("Column names {:?}", df.columns());
    let balanced_dataset = BalancedDataSet::create(df, COLUMN_NAME)?;
    let balanced_sample = balanced_dataset.get_balanced_sample()?;
    for sample in balanced_sample {
        let column = sample.select([COLUMN_NAME])?.select_at_idx(0).unwrap();
        let column_value = match column {
            Column::Scalar(scalar) => scalar.scalar().clone().into_value(),
            Column::Series(series) => series.first().into_value(),
        };
        let column_key = match column_value {
            AnyValue::String(str) => Ok(str),
            _ => Err(PolarsError::ComputeError("Can not cast category to string".into())),
        }?;
        let path = format!("data/ch06/{}.csv", column_key);
        /*let serialize_opts = SerializeOptions::default();
        let serializer = CsvSerializer::new(Arc::new(()), Arc::new(serialize_opts));
        let mut buf = Vec::new();
        serializer.serialize_to_csv(sample, &buf)?;
        let csv_writer = CsvWriter::new(buf);
        csv_writer.
        */
    }
    Ok(())
}
