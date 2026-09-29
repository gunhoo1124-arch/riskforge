import numpy as np
import pandas as pd
import pytest

from riskforge.returns import calculate_log_returns


def test_calculate_log_returns() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])

    result = calculate_log_returns(prices)

    np.testing.assert_allclose(result.to_numpy(), np.log([1.1, 0.9]))
    assert result.name == "log_return"


@pytest.mark.parametrize("prices", [pd.Series([100.0]), pd.Series([100.0, 0.0])])
def test_calculate_log_returns_rejects_invalid_prices(prices: pd.Series) -> None:
    with pytest.raises(ValueError):
        calculate_log_returns(prices)
