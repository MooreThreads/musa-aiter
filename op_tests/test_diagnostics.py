from musa_aiter.diagnostics.env import collect_environment, relevant_env


def test_collect_environment_shape():
    report = collect_environment()
    data = report.to_dict()
    assert "python" in data
    assert "mcc" in data
    assert "mthreads_gmi" in data
    assert "hard_ok" in data


def test_relevant_env_contains_expected_keys():
    env = relevant_env()
    assert "MUSA_AITER_ARCHS" in env
    assert "MUSA_AITER_CACHE_DIR" in env
