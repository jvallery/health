from healthkit.parsing.gpx import parse_gpx_file


def test_parse_gpx_file_distance_and_points():
    route, pts = parse_gpx_file("tests/data/sample_export/workout-routes/route1.gpx")
    assert route["points"] >= 4
    assert 2800.0 <= route["distance_m"] <= 3200.0
    assert pts["d_m"].sum() == route["distance_m"]
