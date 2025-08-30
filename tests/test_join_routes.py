from healthkit.parsing.export_xml import parse_workouts
from healthkit.parsing.gpx import parse_gpx_dir
from healthkit.transform.routes import link_routes


def test_link_routes_matches_sample():
    workouts = parse_workouts("tests/data/sample_export/export.xml")
    routes, _ = parse_gpx_dir("tests/data/sample_export/workout-routes")
    linked = link_routes(workouts, routes)
    assert bool(linked.loc[0, "has_route"]) is True
    assert isinstance(linked.loc[0, "route_id"], str)
