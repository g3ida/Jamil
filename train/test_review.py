"""What a person says of a page is the only truth kept: that it is kept whole, and that nothing else is.

A checked page is written once and read by everything after — the scoring, the
teaching. A page half written, a line of one point, a name that reaches out of
the folder: each would be found weeks later as a score that makes no sense.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from review import Work, handler_for


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "pages").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "pages" / "a_0001.jpg").write_bytes(b"\xff\xd8 not a picture, but bytes \xff\xd9")
    (tmp_path / "labels" / "a_0001.json").write_text(json.dumps({"lines": [{"baseline": [[10, 20], [300, 22]]}, {"baseline": [[10, 60], [300, 61]]}]}))
    (tmp_path / "queue.json").write_text(json.dumps([{"name": "a_0001", "kinds": ["prose"], "book": "a", "page": 1}, {"name": "b_0002", "kinds": ["cover"]}]))
    return tmp_path


def test_a_page_nobody_has_checked_shows_the_network_s_lines(folder: Path) -> None:
    page = Work(folder).page("a_0001")

    assert page["lines"] == [[[10, 20], [300, 22]], [[10, 60], [300, 61]]]
    assert page["status"] is None and page["network"] == 2
    assert page["about"] == {"book": "a", "page": 1}


def test_a_page_with_no_labels_is_shown_empty(folder: Path) -> None:
    assert Work(folder).page("b_0002")["lines"] == []


def test_what_was_said_of_a_page_is_what_it_shows_afterwards(folder: Path) -> None:
    work = Work(folder)

    work.keep("a_0001", {"status": "right", "lines": [[[10, 20], [150, 21], [300, 22]]], "unknown": [[0, 100, 50, 150]], "kinds": ["prose", "old scan"], "network": 2, "changes": 3, "seconds": 41.5})

    page = Work(folder).page("a_0001")
    assert page["status"] == "right"
    assert page["lines"] == [[[10.0, 20.0], [150.0, 21.0], [300.0, 22.0]]]
    assert page["unknown"] == [[0.0, 100.0, 50.0, 150.0]]
    assert page["kinds"] == ["prose", "old scan"]
    assert [entry["status"] for entry in work.listing()] == ["right", None]


def test_a_line_of_one_point_is_refused_and_nothing_is_written(folder: Path) -> None:
    work = Work(folder)

    with pytest.raises(ValueError):
        work.keep("a_0001", {"status": "right", "lines": [[[10, 20]]]})

    assert work.said("a_0001") is None
    assert list((folder / "checked").iterdir()) == []


def test_a_page_is_right_or_set_aside_and_nothing_else(folder: Path) -> None:
    with pytest.raises(ValueError):
        Work(folder).keep("a_0001", {"status": "nearly", "lines": []})


@pytest.fixture
def served(folder: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Work(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _asked(url: str, body: dict | None = None):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as answer:
        return json.loads(answer.read())


def test_a_page_kept_through_the_server_is_on_disk(served: str, folder: Path) -> None:
    _asked(f"{served}/api/page/a_0001", {"status": "set aside", "lines": [], "kinds": ["cover"]})

    assert json.loads((folder / "checked" / "a_0001.json").read_text())["status"] == "set aside"
    assert _asked(f"{served}/api/queue")[0]["status"] == "set aside"


def test_a_name_that_is_not_a_page_of_the_queue_is_not_served(served: str) -> None:
    """The queue is the whole of what may be read or written: not a path, and not a file that happens to be there."""
    for path in ("/api/page/..%2Fqueue", "/api/page/nobody", "/img/..%2F..%2Fqueue.json.jpg"):
        with pytest.raises(urllib.error.HTTPError) as refused:
            _asked(served + path)
        assert refused.value.code == 404


def test_what_is_refused_says_why(served: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as refused:
        _asked(f"{served}/api/page/a_0001", {"status": "right", "lines": [[[1, 2]]]})

    assert refused.value.code == 400
    assert "two points" in json.loads(refused.value.read())["error"]
