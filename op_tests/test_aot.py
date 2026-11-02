from musa_aiter.aot import create_plan, write_plan


def test_create_aot_plan(tmp_path):
    plan = create_plan(
        archs=["mp31"],
        prebuild_level=1,
        output_dir=tmp_path,
        operators=["rmsnorm", "gemm"],
        native_build_enabled=False,
    )
    assert plan.archs == ["mp31"]
    assert plan.operators == ["gemm", "rmsnorm"]
    path = write_plan(plan)
    assert path.exists()
    assert "musa-aiter" in path.read_text()
