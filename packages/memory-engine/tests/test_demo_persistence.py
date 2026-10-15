import json
from pathlib import Path

import pytest

from retrace_memory import demo_persistence


def test_persistence_demo_writes_reopens_and_reads_a_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "demo.sqlite3"

    assert demo_persistence.main([str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert database.exists()
    assert output["memory"]["content"].startswith("Sam recorded")
    assert output["extraction"]["events"][0]["title"] == ("Synthetic project milestone")
    assert output["connections"] == []
    assert output["timeline"]["exact"] == []
    assert len(output["timeline"]["ambiguous"]) == 1
