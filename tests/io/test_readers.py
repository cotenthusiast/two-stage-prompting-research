import pandas as pd
from twoprompt.io.readers import read_raw_questions, read_normalized_questions

def test_read_raw_questions(tmp_path, sample_raw_dataframe):
    raw_csv_path = tmp_path / "raw.csv"
    data = (
        "subject,question,choices,answer\n"
        'computer_security,Which protocol is primarily used to securely browse websites?,"[""FTP"", ""HTTP"", ""HTTPS"", ""SMTP""]",2\n'
        'high_school_physics,What is the SI unit of force?,"[""Joule"", ""Newton"", ""Watt"", ""Pascal""]",1'
    )
    raw_csv_path.write_text(data)
    pd.testing.assert_frame_equal(read_raw_questions("raw.csv",raw_dir=tmp_path),sample_raw_dataframe)

def test_read_normalized_questions(tmp_path, sample_normalized_dataframe):
    normalized_csv_path = tmp_path / "normalized.csv"
    data = (
        "question_id,subject,question_text,choice_a,choice_b,choice_c,choice_d,correct_option,correct_answer_text\n"
        "4865890d7f0efae8,computer_security,Which protocol is primarily used to securely browse websites?,FTP,HTTP,HTTPS,SMTP,C,HTTPS\n"
        "5e9876049bf053f9,high_school_physics,What is the SI unit of force?,Joule,Newton,Watt,Pascal,B,Newton"
    )
    normalized_csv_path.write_text(data)
    pd.testing.assert_frame_equal(read_normalized_questions("normalized.csv", processed_dir = tmp_path),sample_normalized_dataframe)
