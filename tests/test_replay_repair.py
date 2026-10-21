"""The supplementary replay run must not train on its regression holdout."""
import pytest

from scripts.replay_repair import build_replay_rows


def test_replay_fraction_and_holdout_isolation():
    train = [{"instruction": f"triage {i}"} for i in range(225)]
    replay = [{"instruction": f"general {i}"} for i in range(11)]
    regression = [{"instruction": "held out question"}]

    assert len(build_replay_rows(train, replay, regression)) == 236
    with pytest.raises(ValueError, match="holdout"):
        build_replay_rows(train, replay[:-1] + regression, regression)
    with pytest.raises(ValueError, match="between 1% and 5%"):
        build_replay_rows(train, replay * 2, regression)
