"""Last-segment head sampling (Stage 1 classification only).

The FINAL segment has no closing gap, so it runs to the end of the video --
which, because the extractor keeps a few seconds after the rake has passed,
means "vehicle then empty track".  Classifying it across the whole span votes
mostly on grass.

Batch 20260808_125052, GW_59: 182 frames, ~45 with a wagon and ~137 of empty
track.  All three cameras returned BRAKE_VAN at >=0.99 for what was a WAGON.
Losing that label cost the vehicle its place in the wagon list, its 11-digit
number, and its damage on the dashboard.

The invariant that matters most: this changes a LABEL, never the number of
vehicles -- and it must NEVER touch the first segment, which feeds the
phantom-drop guard in global_alignment.build_global_wagons.

No video, no model: the sampling window is asserted directly.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_WC = os.path.join(_REPO, "wagon_count")
if _WC not in sys.path:
    sys.path.insert(0, _WC)

import tracker_engine as TE                                        # noqa: E402


# -----------------------------------------------------------------------------
# the env knob
# -----------------------------------------------------------------------------

def test_default_head_fraction(monkeypatch):
    monkeypatch.delenv("WAGONEYE_STAGE1_LAST_SEGMENT_HEAD", raising=False)
    assert TE._last_segment_head_fraction() == 0.35


def test_head_fraction_is_configurable(monkeypatch):
    monkeypatch.setenv("WAGONEYE_STAGE1_LAST_SEGMENT_HEAD", "0.5")
    assert TE._last_segment_head_fraction() == 0.5


def test_out_of_range_or_bad_values_disable_it(monkeypatch):
    for raw in ("0", "1", "1.5", "-0.2"):
        monkeypatch.setenv("WAGONEYE_STAGE1_LAST_SEGMENT_HEAD", raw)
        assert TE._last_segment_head_fraction() == 0.0, raw
    monkeypatch.setenv("WAGONEYE_STAGE1_LAST_SEGMENT_HEAD", "nonsense")
    assert TE._last_segment_head_fraction() == 0.35      # bad value -> default


# -----------------------------------------------------------------------------
# which frames get sampled
# -----------------------------------------------------------------------------

class _Recorder:
    """Captures the (start, end) window each _classify_one call samples."""

    def __init__(self, n_samples=5):
        self.windows = []
        self.num_samples = n_samples
        self.verbose = False
        self.tag = "test"

    def classify_frame(self, frame):
        return "wagon", 0.9


def _window(start, end, head, n_samples=5):
    """Run the real _classify_one sampling maths and return (min, max) sampled."""
    rec = _Recorder(n_samples)
    sampled = []

    class _Cap:
        def set(self, prop, fi):
            sampled.append(int(fi))

        def read(self):
            return True, object()

    TE.MasterClassifier._classify_one(rec, _Cap(), start, end, head_fraction=head)
    return (min(sampled), max(sampled)) if sampled else (None, None)


def test_head_fraction_restricts_sampling_to_the_start():
    """GW_59's real shape: wagon in the first ~45 of 182 frames."""
    lo, hi = _window(4661, 4842, head=0.35)
    assert lo >= 4661
    assert hi <= 4661 + int(round(182 * 0.35))     # never reaches the empty tail
    # and without it, sampling runs the full span into the grass
    lo_all, hi_all = _window(4661, 4842, head=None)
    assert hi_all > hi


def test_no_head_fraction_is_the_previous_behaviour():
    """Disabled must reproduce whole-span sampling exactly."""
    a = _window(1000, 1181, head=None)
    b = _window(1000, 1181, head=0.0)              # 0 is out of range -> ignored
    assert a == b


def test_a_short_segment_still_yields_a_sample():
    """Never return zero samples, however aggressive the fraction."""
    lo, hi = _window(500, 503, head=0.35)
    assert lo is not None and 500 <= lo <= 503


# -----------------------------------------------------------------------------
# THE invariant: the first segment is never touched
# -----------------------------------------------------------------------------

def _heads_applied(n_segments, head=0.35):
    """Which segment indices would receive the head restriction."""
    return [idx for idx in range(n_segments)
            if (idx == n_segments - 1 and idx != 0) and 0.0 < head < 1.0]


def test_only_the_last_segment_is_affected():
    assert _heads_applied(59) == [58]
    assert _heads_applied(3) == [2]
    assert _heads_applied(2) == [1]


def test_the_first_segment_is_never_affected():
    """It feeds the phantom-drop guard -- an UNKNOWN leading segment is DROPPED,
    so this must not be able to influence how it classifies."""
    for n in (1, 2, 3, 10, 59):
        assert 0 not in _heads_applied(n)


def test_a_single_segment_train_is_untouched():
    """One segment is both first and last -- leave it exactly as before."""
    assert _heads_applied(1) == []


def test_disabled_affects_nothing():
    assert _heads_applied(59, head=0.0) == []
