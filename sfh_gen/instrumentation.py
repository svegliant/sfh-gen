"""
instrumentation.py - what a run of sfh-gen prints, and how long it took.

The log and the results share one stream (stdout), so a job script
redirecting it keeps the two in order; that is also why the modules that run
standalone print rather than log. Everything that decides what that stream
looks like lives here: the logger, the per-step tag, the section banners, and
the timers whose summary closes the run.

Timer and TimingSummary are the group's, with one change: the summary keeps
the order in which the steps ran and takes each as a fraction of the wall
clock rather than of the sum of the parts. Steps 01-03 overlap, so their
durations add up to more than the run lasted and a percentage of the sum
would understate every one of them.
"""

import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

#: Width of the section rules, matching print_config_summary.
RULE_WIDTH = 60


# ── Logger ─────────────────────────────────────────────────────────────────

def setup_logger(verbose: bool) -> logging.Logger:
    """Configure this package's logger only.

    Not the root logger: basicConfig would also set the level of every
    third-party library. Output goes to stdout, where steps 04 and 05 print
    their own diagnostics — those modules also run standalone, so they print
    rather than log — and a redirected job file keeps the two in order.
    """
    logger = logging.getLogger("sfh_gen")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger


class TaggedLogger(logging.LoggerAdapter):
    """Prefix each message with a step tag; the two monomer runs log from
    separate threads."""

    def process(self, msg, kwargs):
        return f"[{self.extra['tag']}] {msg}", kwargs


def banner(logger: logging.Logger, text: str):
    """A step heading in the log."""
    sep = "─" * RULE_WIDTH
    logger.info(sep)
    logger.info(text)
    logger.info(sep)


def block(title: str, lines: List[str], rule: str = "─") -> str:
    """A framed block for the result stream, indented like the config summary.

    Used for output that is read rather than followed: the Hamiltonian itself,
    the timing table. Printed, not logged, so the matrix keeps its columns
    instead of carrying a timestamp on every row. The rule runs to the longest
    line, so a wide matrix is framed rather than overflowing.
    """
    width = max([RULE_WIDTH, len(title)] + [len(ln) for ln in lines])
    sep = "  " + rule * width
    out = [sep, f"  {title}", sep]
    out += [f"  {ln}" if ln else "" for ln in lines]
    out.append(sep)
    return "\n".join(out)


# ── Timing ─────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """A duration as a human reads it: 45.2 s, 2 m 14.3 s, 1 h 12 m 05 s.

    Rounded to a tenth of a second before being split up, so that 59.96 s
    reads as 1 m 00.0 s rather than 60.0 s.
    """
    tenths = round(seconds * 10)
    if tenths < 600:
        return f"{tenths / 10:.1f} s"
    minutes, tenths = divmod(tenths, 600)
    if minutes < 60:
        return f"{minutes} m {tenths / 10:04.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} m {tenths // 10:02d} s"


class Timer:
    """Context manager that logs how long a block took.

    Usage
    -----
    with Timer("CASCI(4,4)"):
        ...
    """

    def __init__(self, label: str, level: Optional[int] = logging.INFO):
        self.label = label
        #: None times the block without logging it -- see TimingSummary.measure.
        self.level = level
        self.elapsed: Optional[float] = None
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        if self.level is not None:
            log.log(self.level, "Starting %s", self.label)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.level is None:
            return
        if exc_type is None:
            log.log(self.level, "Finished %s in %s", self.label,
                    format_duration(self.elapsed))
        else:
            log.log(logging.ERROR, "%s failed after %s", self.label,
                    format_duration(self.elapsed))


class TimingSummary:
    """Durations of the steps of one run, reported in label order.

    A step records when it finishes, so insertion order is finishing order and
    the concurrent steps would land in the table in whichever order they
    happened to end; sorting by label restores the pipeline order, the labels
    being "01 ...", "02 ...". The two monomer steps record from their own
    threads, each writing its own key once, so no lock is needed.
    """

    def __init__(self):
        self._durations: Dict[str, float] = {}

    def record(self, label: str, seconds: float) -> None:
        self._durations[label] = self._durations.get(label, 0.0) + seconds

    def measure(self, label: str) -> "_RecordingTimer":
        """Time a block for the summary without logging anything itself.

        Use this when the code inside already announces its own start and
        finish, so the step is not reported twice under the same label --
        which reads as it having run twice.
        """
        return _RecordingTimer(self, label, level=None)

    @property
    def durations(self) -> List[Tuple[str, float]]:
        return sorted(self._durations.items())

    def format_summary(self, wall: Optional[float] = None,
                       concurrent: bool = False) -> str:
        """The closing timing block.

        `wall` is the wall clock of the whole run, against which the shares
        are taken; without it they are taken against the sum. `concurrent`
        says that some of the steps overlapped, which is why the parts can
        add up to more than the whole.
        """
        if not self._durations:
            return block("Timing", ["no step was run"])

        total = sum(self._durations.values())
        against = wall if wall is not None else total
        width = max(len(label) for label in self._durations) + 2

        lines = []
        for label, seconds in sorted(self._durations.items()):
            share = 100.0 * seconds / against if against > 0 else 0.0
            dots = "." * (width - len(label))
            lines.append(f"{label} {dots} {format_duration(seconds):>12s}"
                         f"   {share:5.1f}%")
        if wall is not None:
            lines += ["", f"{'total wall clock':<{width}} "
                          f"{format_duration(wall):>12s}"]
        if concurrent:
            lines += ["", "Steps that ran concurrently are listed with their own",
                      "durations, which therefore add up to more than the wall",
                      "clock."]
        return block("Timing", lines)


class _RecordingTimer(Timer):
    """A `Timer` that also feeds its elapsed time into a `TimingSummary`."""

    def __init__(self, summary: TimingSummary, label: str,
                 level: Optional[int] = logging.INFO):
        super().__init__(label, level=level)
        self._summary = summary

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        super().__exit__(exc_type, exc_value, exc_traceback)
        if self.elapsed is not None:
            self._summary.record(self.label, self.elapsed)
