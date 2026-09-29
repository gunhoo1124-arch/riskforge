from riskforge import streamlit_components


def test_holdings_cart_uses_component_v2_state_and_drag_events() -> None:
    script = streamlit_components._CART_JS

    assert "setStateValue" in script
    assert "ondragstart" in script
    assert "ondrop" in script
    assert "Streamlit.setComponentValue" not in script
    assert "window.parent.postMessage" not in script


def test_holdings_cart_escapes_user_values_through_dom_properties() -> None:
    script = streamlit_components._CART_JS

    assert "textContent" in script
    assert "innerHTML" not in script
