"""review.py --work DIR: a person checks the lines a network drew, page by page, in a browser.

    python review.py --work pages-to-check --port 8765

A network's lines are nearly right, and nearly right is what a test set may
not be: it would score the next network by how well it agrees with the last.
So every page is looked at by a person, who takes away the lines that are
not lines, draws the ones missed, joins what was broken and cuts what was run
together, and says of the page that it is now right. What they say is the
only thing kept as truth.

DIR holds:

    queue.json            the pages, in the order they are to be seen:
                          [{"name": …, "kinds": […], …}, …]
    pages/NAME.jpg        the picture
    labels/NAME.json      what the network drew: {"lines": [{"baseline": [[x, y], …]}, …]}
                          (the detector's own output), a line from its start
    checked/NAME.json     what the person left, written here:
                          {"status": "right" | "set aside", "lines": [[[x, y], …], …],
                           "unknown": [[x0, y0, x1, y1], …], "kinds": […], …}

A page set aside is one no truth can be had of — blank, a picture, a blur.
`unknown` is ground on a page that nobody vouches for: a stamp over the
text, a margin scribbled full. It is left out of both the teaching and the
scoring, which lets the rest of such a page count.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
# A page's name is a file's name: nothing that could climb out of the folder.
NAME = re.compile(r"^[\w.-]+$")
STATUSES = ("right", "set aside")


class Work:
    """A folder of pages to check, and what has been said of each."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.queue = json.loads((folder / "queue.json").read_text(encoding="utf-8"))
        self.by_name = {entry["name"]: entry for entry in self.queue}
        (folder / "checked").mkdir(exist_ok=True)

    def said(self, name: str) -> dict | None:
        kept = self.folder / "checked" / f"{name}.json"
        return json.loads(kept.read_text(encoding="utf-8")) if kept.exists() else None

    def listing(self) -> list[dict]:
        """The queue, each page with what has been said of it."""
        out = []
        for entry in self.queue:
            said = self.said(entry["name"])
            out.append({"name": entry["name"], "kinds": (said or entry).get("kinds", []), "status": said["status"] if said else None})
        return out

    def page(self, name: str) -> dict:
        """A page as it is to be shown: the person's lines if they have been at it, the network's if not."""
        entry, said = self.by_name[name], self.said(name)
        drawn = self.folder / "labels" / f"{name}.json"
        network = [line["baseline"] for line in json.loads(drawn.read_text(encoding="utf-8"))["lines"]] if drawn.exists() else []
        return {
            "name": name,
            "about": {k: v for k, v in entry.items() if k not in ("name", "kinds")},
            "kinds": (said or entry).get("kinds", []),
            "status": said["status"] if said else None,
            "lines": said["lines"] if said else network,
            "unknown": said.get("unknown", []) if said else [],
            "network": len(network),
        }

    def keep(self, name: str, said: dict) -> None:
        """Writes what a person said of a page, refusing what is not a page's lines."""
        if said.get("status") not in STATUSES:
            raise ValueError(f"a page is {' or '.join(STATUSES)}, not {said.get('status')!r}")
        lines = [[[float(x), float(y)] for x, y in line] for line in said.get("lines", [])]
        if any(len(line) < 2 for line in lines):
            raise ValueError("a line of fewer than two points is no line")
        unknown = [[float(v) for v in box] for box in said.get("unknown", [])]
        if any(len(box) != 4 for box in unknown):
            raise ValueError("ground nobody vouches for is a box of four numbers")
        kept = {
            "status": said["status"], "lines": lines, "unknown": unknown, "kinds": [str(k) for k in said.get("kinds", [])],
            "network": int(said.get("network", 0)), "changes": int(said.get("changes", 0)), "seconds": float(said.get("seconds", 0.0)),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Beside the file and moved over it: a page half written is a page lost.
        target = self.folder / "checked" / f"{name}.json"
        beside = target.with_name(target.name + ".part")
        beside.write_text(json.dumps(kept), encoding="utf-8")
        beside.replace(target)


def handler_for(work: Work):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # a line a request is noise in a terminal someone works in
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, code: int = 200) -> None:
            self._send(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _named(self, prefix: str, suffix: str = "") -> str | None:
            name = self.path.split("?")[0][len(prefix):]
            name = name[: len(name) - len(suffix)] if suffix and name.endswith(suffix) else name
            return name if NAME.match(name) and name in work.by_name else None

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, (HERE / "review.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/queue":
                return self._json(work.listing())
            if path.startswith("/api/page/") and (name := self._named("/api/page/")):
                return self._json(work.page(name))
            if path.startswith("/img/") and (name := self._named("/img/", ".jpg")):
                return self._send(200, (work.folder / "pages" / f"{name}.jpg").read_bytes(), "image/jpeg")
            self._json({"error": "no such thing"}, 404)

        def do_POST(self) -> None:
            if not (self.path.startswith("/api/page/") and (name := self._named("/api/page/"))):
                return self._json({"error": "no such page"}, 404)
            try:
                work.keep(name, json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            except (ValueError, KeyError, TypeError) as wrong:
                return self._json({"error": str(wrong)}, 400)
            self._json({"kept": name})

    return Handler


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--work", required=True, type=Path)
    a.add_argument("--port", type=int, default=8765)
    a.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to check from another machine on the network")
    args = a.parse_args()
    work = Work(args.work)
    print(f"{len(work.queue)} pages to check: http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler_for(work)).serve_forever()


if __name__ == "__main__":
    main()
