"""Execute the Python examples in the documentation so they cannot go stale."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md"]
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_python_examples_run(document, monkeypatch, tmp_path):
    blocks = CODE_BLOCK.findall(document.read_text(encoding="utf-8"))
    assert blocks, f"no python examples found in {document.name}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    namespace = {"__name__": "__docs__"}
    for number, code in enumerate(blocks, start=1):
        try:
            exec(compile(code, f"{document.name}[block {number}]", "exec"), namespace)
        except Exception as exc:
            pytest.fail(f"{document.name} code block {number} failed: {exc!r}\n\n{code}")
