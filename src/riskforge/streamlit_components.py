"""Small Streamlit Custom Components v2 used by the RiskForge dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitAPIException

_CART_HTML = """
<section class="cart" aria-label="Portfolio holdings cart">
  <div class="cartHeader">
    <div>
      <strong>Holdings cart</strong>
      <div class="hint">Drag the handle to reorder, or use the arrow buttons.</div>
    </div>
    <div class="total" aria-live="polite"></div>
  </div>
  <div class="items"></div>
  <div class="addRow">
    <label>Ticker<input class="newTicker" placeholder="AAPL" maxlength="30"></label>
    <label>Amount to invest ($)
      <input class="newAmount" type="number" min="1" step="100" value="10000">
    </label>
    <button class="addButton" type="button">Add to cart</button>
  </div>
  <div class="message" role="status" aria-live="polite"></div>
</section>
"""

_CART_CSS = """
.cart {
  box-sizing: border-box;
  color: var(--st-text-color);
  font-family: var(--st-font);
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius);
  background: var(--st-secondary-background-color);
  padding: 1rem;
}
.cartHeader, .holdingCard, .addRow {
  display: flex;
  align-items: center;
  gap: .75rem;
}
.cartHeader { justify-content: space-between; margin-bottom: .75rem; }
.hint, .message { color: var(--st-text-color); opacity: .72; font-size: .82rem; }
.total { font-weight: 700; color: var(--st-primary-color); white-space: nowrap; }
.items { display: grid; gap: .5rem; }
.holdingCard {
  background: var(--st-background-color);
  border: 1px solid var(--st-border-color-light);
  border-radius: var(--st-base-radius);
  padding: .65rem;
}
.holdingCard.dragOver { outline: 2px solid var(--st-primary-color); }
.dragHandle { cursor: grab; font-size: 1.25rem; user-select: none; }
.holdingCard label, .addRow label { display: grid; gap: .2rem; font-size: .75rem; flex: 1; }
input {
  box-sizing: border-box;
  width: 100%;
  color: var(--st-text-color);
  background: var(--st-background-color);
  border: 1px solid var(--st-widget-border-color);
  border-radius: var(--st-base-radius);
  padding: .5rem .6rem;
  font: inherit;
}
button {
  border: 1px solid var(--st-widget-border-color);
  border-radius: var(--st-button-radius);
  color: var(--st-text-color);
  background: var(--st-background-color);
  padding: .45rem .65rem;
  cursor: pointer;
}
button:hover { border-color: var(--st-primary-color); }
.addButton {
  background: var(--st-primary-color);
  color: var(--st-background-color);
  font-weight: 700;
}
.remove { color: var(--st-red-text-color); }
.moveButtons { display: flex; gap: .25rem; }
.addRow { margin-top: .85rem; align-items: end; }
.message { min-height: 1.2rem; margin-top: .4rem; }
@media (max-width: 700px) {
  .holdingCard, .addRow { align-items: stretch; flex-direction: column; }
  .dragHandle { display: none; }
}
"""

_CART_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component
  const itemsRoot = parentElement.querySelector(".items")
  const totalRoot = parentElement.querySelector(".total")
  const messageRoot = parentElement.querySelector(".message")
  const tickerInput = parentElement.querySelector(".newTicker")
  const amountInput = parentElement.querySelector(".newAmount")
  const addButton = parentElement.querySelector(".addButton")
  if (!itemsRoot || !totalRoot || !messageRoot || !tickerInput || !amountInput || !addButton) return

  const maxHoldings = Number(data?.maxHoldings ?? 10)
  let holdings = Array.isArray(data?.holdings)
    ? data.holdings.map(item => ({
        ticker: String(item?.ticker ?? "").toUpperCase(),
        amount: Number(item?.amount ?? 0),
      }))
    : []

  const money = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  })

  function emit(next) {
    holdings = next
    render()
    setStateValue("holdings", next)
  }

  function move(from, to) {
    if (to < 0 || to >= holdings.length || from === to) return
    const next = holdings.slice()
    const [item] = next.splice(from, 1)
    next.splice(to, 0, item)
    emit(next)
  }

  function render() {
    itemsRoot.replaceChildren()
    holdings.forEach((holding, index) => {
      const card = document.createElement("div")
      card.className = "holdingCard"
      card.dataset.index = String(index)

      const handle = document.createElement("span")
      handle.className = "dragHandle"
      handle.textContent = "⠿"
      handle.title = "Drag to reorder"
      handle.setAttribute("aria-label", `Drag ${holding.ticker || "holding"} to reorder`)
      handle.draggable = true
      handle.ondragstart = event => {
        event.dataTransfer.setData("text/plain", String(index))
        event.dataTransfer.effectAllowed = "move"
      }
      card.ondragover = event => {
        event.preventDefault()
        card.classList.add("dragOver")
      }
      card.ondragleave = () => card.classList.remove("dragOver")
      card.ondrop = event => {
        event.preventDefault()
        card.classList.remove("dragOver")
        move(Number(event.dataTransfer.getData("text/plain")), index)
      }

      const tickerLabel = document.createElement("label")
      tickerLabel.textContent = "Ticker"
      const ticker = document.createElement("input")
      ticker.value = holding.ticker
      ticker.maxLength = 30
      ticker.setAttribute("aria-label", `Ticker for holding ${index + 1}`)
      ticker.onchange = event => {
        const next = holdings.slice()
        next[index] = { ...next[index], ticker: event.target.value.trim().toUpperCase() }
        emit(next)
      }
      tickerLabel.appendChild(ticker)

      const amountLabel = document.createElement("label")
      amountLabel.textContent = "Amount to invest ($)"
      const amount = document.createElement("input")
      amount.type = "number"
      amount.min = "1"
      amount.step = "100"
      amount.value = String(holding.amount)
      const amountName = holding.ticker || `holding ${index + 1}`
      amount.setAttribute("aria-label", `Investment amount for ${amountName}`)
      amount.onchange = event => {
        const next = holdings.slice()
        next[index] = { ...next[index], amount: Number(event.target.value) }
        emit(next)
      }
      amountLabel.appendChild(amount)

      const moveButtons = document.createElement("div")
      moveButtons.className = "moveButtons"
      const up = document.createElement("button")
      up.type = "button"
      up.textContent = "↑"
      up.disabled = index === 0
      up.setAttribute("aria-label", `Move ${holding.ticker || "holding"} up`)
      up.onclick = () => move(index, index - 1)
      const down = document.createElement("button")
      down.type = "button"
      down.textContent = "↓"
      down.disabled = index === holdings.length - 1
      down.setAttribute("aria-label", `Move ${holding.ticker || "holding"} down`)
      down.onclick = () => move(index, index + 1)
      moveButtons.append(up, down)

      const remove = document.createElement("button")
      remove.type = "button"
      remove.className = "remove"
      remove.textContent = "Remove"
      remove.setAttribute("aria-label", `Remove ${holding.ticker || "holding"}`)
      remove.onclick = () => emit(holdings.filter((_, itemIndex) => itemIndex !== index))

      card.append(handle, tickerLabel, amountLabel, moveButtons, remove)
      itemsRoot.appendChild(card)
    })
    const total = holdings.reduce(
      (sum, item) => sum + (Number.isFinite(item.amount) ? item.amount : 0),
      0,
    )
    totalRoot.textContent = `Cart total: ${money.format(total)}`
    messageRoot.textContent = holdings.length >= maxHoldings
      ? `Maximum of ${maxHoldings} holdings reached.`
      : `${holdings.length} of ${maxHoldings} holdings in cart.`
    addButton.disabled = holdings.length >= maxHoldings
  }

  addButton.onclick = () => {
    const ticker = tickerInput.value.trim().toUpperCase()
    const amount = Number(amountInput.value)
    if (!ticker) {
      messageRoot.textContent = "Enter a ticker before adding it."
      return
    }
    if (!Number.isFinite(amount) || amount <= 0) {
      messageRoot.textContent = "Enter a positive dollar amount."
      return
    }
    if (holdings.some(item => item.ticker === ticker)) {
      messageRoot.textContent = `${ticker} is already in the cart.`
      return
    }
    if (holdings.length >= maxHoldings) return
    tickerInput.value = ""
    emit([...holdings, { ticker, amount }])
  }
  tickerInput.onkeydown = event => {
    if (event.key === "Enter") {
      event.preventDefault()
      addButton.click()
    }
  }
  render()
}
"""

_HOLDINGS_CART = st.components.v2.component(
    "riskforge_holdings_cart",
    html=_CART_HTML,
    css=_CART_CSS,
    js=_CART_JS,
)


def holdings_cart(
    initial_holdings: Sequence[Mapping[str, Any]],
    *,
    key: str,
    max_holdings: int = 10,
) -> list[dict[str, object]]:
    """Render a draggable holdings cart and return its current ticker-dollar rows."""
    global _HOLDINGS_CART

    normalized = [
        {"ticker": str(item.get("ticker", "")), "amount": float(item.get("amount", 0))}
        for item in initial_holdings
    ]
    component_state = st.session_state.get(key, {})
    current = component_state.get("holdings", normalized)
    mount_options = {
        "key": key,
        "data": {"holdings": current, "maxHoldings": max_holdings},
        "default": {"holdings": current},
        "on_holdings_change": lambda: None,
        "width": "stretch",
    }
    try:
        result = _HOLDINGS_CART(**mount_options)
    except StreamlitAPIException as exc:
        if "is not registered" not in str(exc):
            raise
        _HOLDINGS_CART = st.components.v2.component(
            "riskforge_holdings_cart",
            html=_CART_HTML,
            css=_CART_CSS,
            js=_CART_JS,
        )
        result = _HOLDINGS_CART(**mount_options)
    holdings = getattr(result, "holdings", current)
    if not isinstance(holdings, list):
        return normalized
    return [dict(item) for item in holdings if isinstance(item, Mapping)]
