from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from pathlib import Path


def test_project_data_path_resolves_installed_data_files(tmp_path):
    from musa_aiter._paths import resolve_project_data_path

    package_root = tmp_path / "site-packages"
    install_prefix = tmp_path / "venv"
    expected = install_prefix / "csrc" / "smoke" / "add.mu"
    expected.parent.mkdir(parents=True)
    expected.write_text("// test\n", encoding="utf-8")

    resolved = resolve_project_data_path(
        "csrc",
        "smoke",
        "add.mu",
        package_root=package_root,
        install_prefix=install_prefix,
    )

    assert resolved == expected


def test_pyproject_includes_native_sources_and_example_configs_in_distributions():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    data_files = pyproject["tool"]["setuptools"]["data-files"]

    assert "csrc/smoke/*.mu" in data_files["csrc/smoke"]
    assert "csrc/kvcache/*.mu" in data_files["csrc/kvcache"]
    assert "csrc/ipc/*.cpp" in data_files["csrc/ipc"]
    assert "configs/tuned/*.csv" in data_files["configs/tuned"]


def test_pyproject_release_metadata_declares_safe_install_extras_and_notices():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    extras = project["optional-dependencies"]

    assert "torch>=2.0" in extras["cpu"]
    assert "vllm-reference" in extras
    assert "vllm" not in extras
    assert project["license"] == "MIT"
    assert "ThirdPartyNotices.txt" in project["license-files"]
    assert "include ThirdPartyNotices.txt" in Path("MANIFEST.in").read_text(encoding="utf-8")


def test_install_docs_keep_musa_and_cpu_paths_separate():
    readme = Path("README.md").read_text(encoding="utf-8")
    english_readme = Path("README.en.md").read_text(encoding="utf-8")
    install_guide = Path("docs/deploy/package-install.md").read_text(encoding="utf-8")

    for document in (readme, install_guide):
        assert "python3 -m pip install --no-deps musa-aiter" in document
        assert 'python3 -m pip install "musa-aiter[cpu]"' in document
        assert "vendor-matched" in document or "版本匹配" in document

    assert "vendor-matched" in english_readme
    assert "python3 -m pip install torch\npython3 -m pip install musa-aiter" not in readme
