import numpy as np

from riskforge.visualization import build_fan_chart


def test_build_fan_chart_returns_interactive_figure() -> None:
    paths = np.array(
        [
            [100.0, 101.0, 102.0],
            [100.0, 99.0, 98.0],
            [100.0, 103.0, 101.0],
        ]
    )

    figure = build_fan_chart(paths, "SPY", sample_paths=2)

    assert figure.layout.title.text.startswith("SPY")
    assert len(figure.data) == 7
