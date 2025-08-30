from pathlib import Path

from click.testing import CliRunner

from healthkit.cli import main


def test_cli_convert_end_to_end(tmp_path: Path):
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        [
            "convert",
            "--xml",
            "tests/data/sample_export/export.xml",
            "--routes",
            "tests/data/sample_export/workout-routes",
            "--ecg",
            "tests/data/sample_export/electrocardiograms",
            "--out",
            str(out_dir),
            "--tz",
            "America/Denver",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "workouts.parquet").exists()
    assert (out_dir / "routes.parquet").exists()
    assert (out_dir / "workout_stats.parquet").exists()
