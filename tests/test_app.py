import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_empty_workspace_navigation(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/default.yaml", tmp_path / "configs/default.yaml")
    (tmp_path / "data/metadata").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/metadata/recordings.csv", tmp_path / "data/metadata/recordings.csv")
    monkeypatch.setenv("NEUROQ_ROOT", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "NeuroQ-Connect"
    for page in ["Datasets", "Analysis", "Experiments", "Team", "Software check", "Workspace"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Failed page: {page}"
    app.button[0].click().run()
    assert app.sidebar.radio[0].value == "Datasets"
    assert not app.exception
