import pandas as pd
from twoprompt.io.writers import write_normalized_questions

def test_write_normalized_questions(sample_raw_dataframe, sample_normalized_dataframe, tmp_path):
    raw_csv_path = tmp_path / "raw.csv"
    normalized_csv_path = tmp_path / "normalized.csv"

    sample_raw_dataframe.to_csv(raw_csv_path, index=False)

    write_normalized_questions(
        raw_questions_path=raw_csv_path,
        normalized_questions_path=normalized_csv_path,
    )

    actual_df = pd.read_csv(normalized_csv_path)
    pd.testing.assert_frame_equal(actual_df, sample_normalized_dataframe)