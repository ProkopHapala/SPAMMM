"""
review.py — L1 agent-review artifacts for pytest (.out curated, .log trace).

.out  — evaluation packet: intent, tables, metrics, agent checklist (read first)
.log  — verbose execution trace (read when .out shows a problem)
stdout — progress + REVIEW: path pointers (never filter when running tests)

See doc/TEST_DESIGN.md and skill:running-tests.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from spammm.globals import debug_summarize_array


class ReviewSession:
    """Per-test review writer. Use via make_review fixture in conftest."""

    def __init__(self, outdir: str, test_name: str, enabled: bool = True):
        self.outdir = outdir
        self.test_name = test_name
        self.enabled = enabled
        self._out: list[str] = []
        self._log: list[str] = []
        self._t0 = time.time()

    @property
    def active(self) -> bool:
        return self.enabled

    def out(self, line: str = '') -> None:
        if self.enabled:
            self._out.append(line)

    def log(self, line: str = '') -> None:
        if self.enabled:
            self._log.append(line)

    def out_section(self, title: str) -> None:
        self.out(f'\n## {title}\n')

    def log_section(self, title: str) -> None:
        self.log(f'\n--- {title} ---')

    def array_summary(self, name: str, x: Any, *, channel: str = 'out') -> None:
        line = f'{name}: {debug_summarize_array(x)}'
        getattr(self, channel)(line)

    def graph_table(self, graph, *, channel: str = 'out', **kwargs) -> None:
        text = graph.format_table(**kwargs)
        getattr(self, channel)(text)

    def checklist(self, *items: str) -> None:
        self.out_section('Agent checklist')
        for i, item in enumerate(items, 1):
            self.out(f'{i}. {item}')

    def finish(self) -> Optional[str]:
        """Write .out/.log files; print REVIEW lines to stdout. Returns .out path."""
        if not self.enabled:
            return None
        os.makedirs(self.outdir, exist_ok=True)
        elapsed = time.time() - self._t0
        header = f'# {self.test_name}\n# elapsed={elapsed:.3f}s\n'
        out_path = os.path.join(self.outdir, f'{self.test_name}.out')
        log_path = os.path.join(self.outdir, f'{self.test_name}.log')
        with open(out_path, 'w') as f:
            f.write(header + '\n'.join(self._out) + '\n')
        with open(log_path, 'w') as f:
            f.write(header + f'# trace for {self.test_name}\n' + '\n'.join(self._log) + '\n')
        print(f'REVIEW: {out_path}', flush=True)
        print(f'REVIEW: {log_path}', flush=True)
        return out_path


class _LogTee:
    """Tee debug_print / trace lines into active ReviewSession .log buffers."""

    def __init__(self):
        self._sessions: list[ReviewSession] = []

    def register(self, session: ReviewSession) -> None:
        if session.enabled:
            self._sessions.append(session)

    def clear(self) -> None:
        self._sessions.clear()

    def write(self, line: str) -> None:
        for s in self._sessions:
            s.log(line.rstrip('\n'))


_log_tee = _LogTee()


def get_log_tee() -> _LogTee:
    return _log_tee


class review_trace:
    """Context manager: register session for log tee during test body."""

    def __init__(self, session: ReviewSession):
        self.session = session

    def __enter__(self):
        if self.session.enabled:
            _log_tee.register(self.session)
            self.session.log_section(f'start {self.session.test_name}')
        return self.session

    def __exit__(self, *exc):
        if self.session.enabled:
            if exc[0]:
                self.session.log(f'EXCEPTION: {exc[0]!r}')
            self.session.log_section(f'end {self.session.test_name}')
        return False
