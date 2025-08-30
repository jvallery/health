from healthkit.parsing.ecg_csv import parse_ecg_csv


def test_parse_ecg_csv_sample():
    row, _ = parse_ecg_csv("tests/data/sample_export/electrocardiograms/ecg_sample.csv")
    assert row["ecg_id"].startswith("ecg_") or row["ecg_id"]
    assert row["mean_hr_bpm"] == 72.0
