import dashboard


def test_strategy_mode_uses_ml_only_for_ml_modes() -> None:
    assert dashboard._strategy_mode_uses_ml("ml") is True
    assert dashboard._strategy_mode_uses_ml("hybrid") is True
    assert dashboard._strategy_mode_uses_ml("mean_reversion") is False
    assert dashboard._strategy_mode_uses_ml("breakout") is False
    assert dashboard._strategy_mode_uses_ml(None) is False
