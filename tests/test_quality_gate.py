"""tools/check.py's coverage floor must mean what it says."""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_coverage_floor_is_not_met_by_rounding():
    """coverage compares the total ROUNDED to its report precision with the
    floor; at the default precision 0, a measured 89.93% passed a 90% floor."""
    from coverage.results import should_fail_under
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        precision = tomllib.load(fh)["tool"]["coverage"]["report"]["precision"]
    assert precision >= 2
    assert should_fail_under(89.93, 90, precision) is True
    assert should_fail_under(90.0, 90, precision) is False
