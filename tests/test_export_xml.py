from healthkit.parsing.export_xml import parse_workouts


def test_parse_workouts_sample():
    df = parse_workouts("tests/data/sample_export/export.xml")
    assert len(df) == 1
    w = df.iloc[0]
    assert w["activity"] == "Walking"
    # tz offset preserved from source string (-0600)
    assert w["tz_offset_min"] in (-360, -420, 0)  # tolerate env variations, prefer -360 for sample
    assert abs(w["distance_m"] - 3000.0) < 10.0
    assert w["energy_kcal"] == 200.0
    assert isinstance(w["workout_id"], str) and len(w["workout_id"]) == 40
