import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))
from benchmark_corpus import make_benchmark_corpus, validate_corpus


def test_fixed_corpus_shape_and_hashes_are_deterministic():
    first = make_benchmark_corpus()
    second = make_benchmark_corpus()
    validate_corpus(first)
    validate_corpus(second)
    assert len(first) == 28
    assert sum(len(case.gap_windows) for case in first) == 131
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert [case.digest() for case in first] == [case.digest() for case in second]


def test_speed_range_contains_normal_and_extreme_cases():
    cases = make_benchmark_corpus()
    speeds = {round(case.max_abs_speed_rpm) for case in cases}
    assert 100 in speeds
    assert 500 in speeds
    assert 1500 in speeds
    assert 3000 in speeds
    assert 6000 in speeds
    assert 10000 in speeds
