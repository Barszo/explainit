"""Execute the Python examples in the documentation so they cannot go stale.

Code blocks in a document share one namespace and run in order. When a
``python`` block is immediately followed by a ``text`` block, the text block is
the expected standard output of the code.
"""

import contextlib
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"```python\n(?P<code>.*?)```\n(?:\n```text\n(?P<output>.*?)```)?", re.DOTALL)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_python_examples_run(document, monkeypatch, tmp_path):
    blocks = list(BLOCK.finditer(document.read_text(encoding="utf-8")))
    assert blocks, f"no python examples found in {document.name}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    namespace = {"__name__": "__docs__"}
    for number, block in enumerate(blocks, start=1):
        code, expected = block["code"], block["output"]
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, f"{document.name}[block {number}]", "exec"), namespace)
        except Exception as exc:
            pytest.fail(f"{document.name} code block {number} failed: {exc!r}\n\n{code}")
        if expected is not None:
            assert stdout.getvalue() == expected, f"{document.name} code block {number}"
