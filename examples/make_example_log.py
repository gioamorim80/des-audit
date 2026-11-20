"""Write ``example_log.csv``: 12,000 offer decisions from a made-up store, with a known truth so
the commands' output can be read against it.

- Each decision is a chat with a shopper drawn with replacement from 9,000, so 6,651 shoppers
  appear and many chat more than once (``shopper_id``). A shopper is price-sensitive (25%),
  shipping-sensitive (20%) or neither.
- Context: ``ctx_cart_value`` (lognormal around $80), ``ctx_pages_viewed`` (Poisson, mean 4),
  ``ctx_returning`` (40%). Purchase probability with no offer: 0.06 + 0.015 × pages (capped at 10)
  + 0.05 if returning, about 0.14 on average.
- Offers: 10% off adds 10 points of purchase probability for a price-sensitive shopper and 2 for
  anyone else; free shipping adds 10 for a shipping-sensitive shopper and 2 for anyone else.
- Judgments read from each chat: ``j_price_is_obstacle`` and ``j_shipping_cost_objection`` are high
  for the matching kind (Beta(6, 2) against Beta(1, 6) otherwise); ``j_will_buy_without_offer`` is
  the no-offer probability plus noise, so it tracks who buys, not whom an offer moves. 1% of
  ``j_shipping_cost_objection`` readings are missing.
- The logging policy offers 10% off more often to shoppers who viewed more pages (0.2 + 0.04 per
  page up to 10), splits the rest 60/40 between no offer and free shipping, and gives every arm at
  least 10%. The propensity of the offer made is logged.
- Economics: a sale earns 70% of the cart (``revenue``); ``cost`` is 10% of the cart for a sale
  with 10% off and $8 for a sale with free shipping.

    python examples/make_example_log.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("no_offer", "discount_10", "free_shipping")
N_DECISIONS, N_SHOPPERS, SEED = 12_000, 9_000, 2026


def propensities(pages: np.ndarray) -> np.ndarray:
    """Return the logging policy's (n, 3) arm probabilities, every arm at least 0.1."""
    discount = 0.2 + 0.04 * np.minimum(pages, 10)
    p = np.column_stack([(1 - discount) * 0.6, discount, (1 - discount) * 0.4])
    p = np.maximum(p, 0.1)
    return p / p.sum(axis=1, keepdims=True)


def make_log(n: int = N_DECISIONS, shoppers: int = N_SHOPPERS, seed: int = SEED) -> pd.DataFrame:
    """Return the example log."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(["price", "shipping", "neither"], size=shoppers, p=[0.25, 0.20, 0.55])
    returning = rng.random(shoppers) < 0.4
    who = rng.integers(0, shoppers, size=n)
    k, back = kind[who], returning[who]
    cart = np.round(rng.lognormal(np.log(80), 0.5, size=n), 2)
    pages = rng.poisson(4, size=n)
    p0 = 0.06 + 0.015 * np.minimum(pages, 10) + 0.05 * back
    p = np.column_stack([p0, p0 + np.where(k == "price", 0.10, 0.02), p0 + np.where(k == "shipping", 0.10, 0.02)])
    j_price = np.where(k == "price", rng.beta(6, 2, n), rng.beta(1, 6, n))
    j_ship = np.where(k == "shipping", rng.beta(6, 2, n), rng.beta(1, 6, n))
    j_buy = np.clip(p0 + rng.normal(0, 0.08, n), 0, 1)
    j_ship[rng.random(n) < 0.01] = np.nan
    pi = propensities(pages)
    arm = np.minimum((rng.random(n)[:, None] > pi.cumsum(axis=1)).sum(axis=1), 2)
    bought = rng.random(n) < p[np.arange(n), arm]
    cost = np.where(arm == 1, 0.1 * cart, np.where(arm == 2, 8.0, 0.0)) * bought
    return pd.DataFrame(
        {
            "decision_id": [f"d{i:05d}" for i in range(n)],
            "shopper_id": [f"s{i:04d}" for i in who],
            "action": np.array(ARMS)[arm],
            "propensity": np.round(pi[np.arange(n), arm], 4),
            "outcome": bought.astype(int),
            "revenue": np.round(0.7 * cart * bought, 2),
            "cost": np.round(cost, 2),
            "ctx_cart_value": cart,
            "ctx_pages_viewed": pages,
            "ctx_returning": back.astype(int),
            "j_price_is_obstacle": np.round(j_price, 2),
            "j_shipping_cost_objection": np.round(j_ship, 2),
            "j_will_buy_without_offer": np.round(j_buy, 2),
        }
    )


if __name__ == "__main__":
    out = Path(__file__).with_name("example_log.csv")
    make_log().to_csv(out, index=False)
    print(f"wrote {out}")
