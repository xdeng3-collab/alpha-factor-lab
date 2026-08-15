import pandas as pd
import pytest

from alpha.panel import Panel, synthetic_panel


def test_unsorted_or_duplicated_timestamps_are_rejected():
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-01"]))
    with pytest.raises(ValueError):
        Panel(close=frame)
    dup = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-01"]))
    with pytest.raises(ValueError):
        Panel(close=dup)


def test_volume_must_align_with_close():
    panel = synthetic_panel(5, 30, seed=1)
    with pytest.raises(ValueError):
        Panel(close=panel.close, volume=panel.volume.iloc[:-1])


def test_forward_return_looks_exactly_one_step_ahead():
    panel = synthetic_panel(3, 10, seed=2)
    forward = panel.forward_return(1)
    expected = panel.close.iloc[1, 0] / panel.close.iloc[0, 0] - 1.0
    assert forward.iloc[0, 0] == pytest.approx(expected)
    assert forward.iloc[-1].isna().all()  # nothing after the last row


def test_seed_determines_everything():
    a, b = synthetic_panel(5, 50, seed=9), synthetic_panel(5, 50, seed=9)
    pd.testing.assert_frame_equal(a.close, b.close)
