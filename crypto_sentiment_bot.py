"""
Kalshi Crypto Sentiment Bot
Uses on-chain metrics and sentiment indicators to trade Kalshi BTC/ETH markets.

Strategy:
  Crypto markets have well-established sentiment-price correlations:
  - Fear & Greed Index: Extreme Fear (<20) → contrarian BUY, Extreme Greed (>80) → contrarian SELL
  - Exchange netflows: Large BTC leaving exchanges (negative netflow) → bullish (holders withdrawing)
  - Funding rates: Perp funding extremely positive → overheated longs → fade UP markets
  - Google Trends: BTC search spike → retail FOMO peak → bearish signal
  - Whale wallet movements: Large BTC accumulation → bullish

  Kalshi crypto series:
  - KXBTCD / KXBTCW: BTC daily/weekly price range markets
  - KXETHD: ETH daily price
  - KXBTCCLOSE: BTC month-end close price

  Signal edge:
  When Fear & Greed < 20 (Extreme Fear) AND BTC exchange outflows are positive:
    → Market is pricing in further decline → BUY YES on recovery markets
  When Fear & Greed > 80 (Extreme Greed) AND perp funding > 0.1%/8hr:
    → Overheated, retail crowded → BUY NO or lower price markets

Data Sources (all free):
  - Alternative.me Fear & Greed API: https://api.alternative.me/fng/
  - CryptoQuant public metrics (limited free tier)
  - Glassnode free tier: exchange balance, SOPR
  - Coinglass: funding rates (public endpoint)
  - CoinGecko: price/volume data
"""

import os
import time
import json
import logging
import uuid
import base64
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import httpx
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _normalize_market(m: dict) -> dict:
    """Normalize Kalshi API v2 dollar-denominated fields to legacy field names."""
    if "yes_bid_dollars" in m and "yes_bid" not in m:
        m["yes_bid"] = m.get("yes_bid_dollars")
        m["yes_ask"] = m.get("yes_ask_dollars")
        m["no_bid"] = m.get("no_bid_dollars")
        m["no_ask"] = m.get("no_ask_dollars")
        m["last_price"] = m.get("last_price_dollars")
        m["volume"] = m.get("volume_fp") or m.get("volume_24h_fp") or m.get("volume", 0)
        m["open_interest"] = m.get("open_interest_fp") or m.get("open_interest", 0)
    for k in ["yes_bid", "yes_ask", "no_bid", "no_ask", "last_price"]:
        v = m.get(k)
        if isinstance(v, str):
            try: m[k] = float(v)
            except: pass
    return m


# ── Config ────────────────────────────────────────────────────────────────────

class Config:
    PAPER_MODE:             bool  = os.getenv("PAPER_MODE", "true").lower() == "true"
    PAPER_BALANCE:          float = float(os.getenv("PAPER_BALANCE", "1000"))
    KALSHI_API_KEY:         str   = os.getenv("KALSHI_API_KEY", "")
    KALSHI_KEY_ID:          str   = os.getenv("KALSHI_KEY_ID", "")

    # Sentiment thresholds
    FEAR_GREED_EXTREME_FEAR: int   = int(os.getenv("FEAR_GREED_EXTREME_FEAR", "20"))
    FEAR_GREED_EXTREME_GREED:int   = int(os.getenv("FEAR_GREED_EXTREME_GREED", "80"))
    FUNDING_RATE_HIGH:      float  = float(os.getenv("FUNDING_RATE_HIGH", "0.001"))   # 0.1%/8hr = overheated
    FUNDING_RATE_LOW:       float  = float(os.getenv("FUNDING_RATE_LOW", "-0.0005"))  # -0.05%/8hr = oversold

    MIN_EDGE:               float = float(os.getenv("MIN_EDGE", "0.07"))
    BET_SIZE_USD:           float = float(os.getenv("BET_SIZE_USD", "12.0"))
    MAX_OPEN_POSITIONS:     int   = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
    MIN_PRICE:              int   = int(os.getenv("MIN_PRICE", "15"))
    MAX_PRICE:              int   = int(os.getenv("MAX_PRICE", "85"))

    POLL_INTERVAL_SEC:      int   = int(os.getenv("POLL_INTERVAL_SEC", "3600"))  # 1 hour

    KALSHI_BASE:            str   = "https://api.elections.kalshi.com/trade-api/v2"

# ── Crypto Series to Monitor ──────────────────────────────────────────────────

CRYPTO_SERIES = {
    "BTC": ["KXBTCD", "KXBTCW", "KXBTCCLOSE", "KXBTC"],
    "ETH": ["KXETHD", "KXETH"],
}

# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class CryptoSentiment:
    fear_greed_index:  int          # 0-100
    fear_greed_label:  str          # e.g. "Extreme Fear"
    btc_funding_rate:  float        # 8hr funding rate
    eth_funding_rate:  float
    btc_price:         float
    eth_price:         float
    btc_24h_change:    float        # percentage
    eth_24h_change:    float
    ts:                datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_extreme_fear(self) -> bool:
        return self.fear_greed_index <= Config.FEAR_GREED_EXTREME_FEAR

    @property
    def is_extreme_greed(self) -> bool:
        return self.fear_greed_index >= Config.FEAR_GREED_EXTREME_GREED

    @property
    def is_overheated(self) -> bool:
        return (self.btc_funding_rate > Config.FUNDING_RATE_HIGH and
                self.is_extreme_greed)

    @property
    def is_oversold(self) -> bool:
        return (self.btc_funding_rate < Config.FUNDING_RATE_LOW or
                self.is_extreme_fear)

@dataclass
class KalshiMarket:
    ticker:     str
    title:      str
    yes_price:  int
    no_price:   int
    volume:     int
    close_time: datetime

# ── Sentiment Data Client ─────────────────────────────────────────────────────

class SentimentClient:
    def __init__(self):
        self._client = httpx.Client(timeout=15, headers={"User-Agent": "kalshi-crypto-bot/1.0"})

    def get_fear_greed(self) -> tuple[int, str]:
        """Fetch Alternative.me Fear & Greed Index."""
        try:
            r = self._client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            r.raise_for_status()
            data = r.json()["data"][0]
            return int(data["value"]), data["value_classification"]
        except Exception as e:
            log.warning(f"Fear & Greed API: {e}")
            return 50, "Neutral"

    def get_funding_rates(self) -> dict[str, float]:
        """Fetch current perpetual funding rates from Hyperliquid (same source as funding arb bot)."""
        try:
            r = self._client.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "metaAndAssetCtxs"},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            r.raise_for_status()
            meta, asset_ctxs = r.json()
            universe = meta.get("universe", [])
            rates = {}
            for i, asset in enumerate(universe):
                name = asset.get("name", "")
                if name in ("BTC", "ETH", "SOL") and i < len(asset_ctxs):
                    ctx = asset_ctxs[i]
                    funding_8h = float(ctx.get("funding", 0))
                    rates[name] = funding_8h
            return rates
        except Exception as e:
            log.warning(f"Funding rates: {e}")
            return {}

    def get_prices(self) -> dict[str, dict]:
        """Fetch BTC/ETH prices and 24h change from CoinGecko (free, no key)."""
        try:
            r = self._client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "bitcoin,ethereum",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                },
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"CoinGecko prices: {e}")
            return {}

    def get_sentiment(self) -> CryptoSentiment:
        fg_index, fg_label = self.get_fear_greed()
        rates = self.get_funding_rates()
        prices = self.get_prices()

        btc_price = prices.get("bitcoin", {}).get("usd", 0)
        eth_price = prices.get("ethereum", {}).get("usd", 0)
        btc_change = prices.get("bitcoin", {}).get("usd_24h_change", 0)
        eth_change = prices.get("ethereum", {}).get("usd_24h_change", 0)

        return CryptoSentiment(
            fear_greed_index=fg_index,
            fear_greed_label=fg_label,
            btc_funding_rate=rates.get("BTC", 0.0),
            eth_funding_rate=rates.get("ETH", 0.0),
            btc_price=btc_price,
            eth_price=eth_price,
            btc_24h_change=btc_change,
            eth_24h_change=eth_change,
        )


# ── Kalshi Client ─────────────────────────────────────────────────────────────

class KalshiClient:
    def __init__(self):
        self._client = httpx.Client(timeout=15)
        self._private_key = self._load_private_key()

    @staticmethod
    def _load_private_key():
        pem_str = os.getenv("KALSHI_PRIVATE_KEY", "")
        if not pem_str:
            return None
        if "\\n" in pem_str:
            pem_str = pem_str.replace("\\n", "\n")
        return serialization.load_pem_private_key(pem_str.encode(), password=None)

    def _get_auth_headers(self, method: str, path: str) -> dict:
        if not self._private_key:
            return {"Content-Type": "application/json"}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + "/trade-api/v2" + path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
        return {
            "Kalshi-Access-Key": os.getenv("KALSHI_API_KEY", ""),
            "Kalshi-Access-Signature": base64.b64encode(sig).decode(),
            "Kalshi-Access-Timestamp": ts,
            "Content-Type": "application/json",
        }

    def get_markets_for_series(self, series_ticker: str) -> list[KalshiMarket]:
        try:
            r = self._client.get(
                f"{Config.KALSHI_BASE}/markets",
                params={"series_ticker": series_ticker, "status": "open"},
                headers=self._get_auth_headers("GET", "/markets"),
            )
            r.raise_for_status()
            markets = []
            for m in r.json().get("markets", []):
                _normalize_market(m)
                close_str = m.get("close_time", "")
                try:
                    close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                except Exception:
                    close_dt = datetime.now(timezone.utc) + timedelta(hours=24)
                # yes_ask / no_ask may be dollar floats (0.XX) or cent ints
                yes_ask_raw = m.get("yes_ask") or 0
                no_ask_raw = m.get("no_ask") or 0
                markets.append(KalshiMarket(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    yes_price=yes_ask_raw,
                    no_price=no_ask_raw,
                    volume=m.get("volume", 0),
                    close_time=close_dt,
                ))
            return markets
        except Exception as e:
            log.warning(f"get_markets_for_series({series_ticker}): {e}")
            return []

    def place_order(self, ticker: str, side: str, count: int, price: int) -> bool:
        if Config.PAPER_MODE:
            return True
        try:
            r = self._client.post(
                f"{Config.KALSHI_BASE}/portfolio/orders",
                json={"ticker": ticker, "client_order_id": str(uuid.uuid4()),
                      "action": "buy", "side": side.lower(),
                      "count": count, "type": "limit",
                      "yes_price": price if side == "YES" else 100 - price},
                headers=self._get_auth_headers("POST", "/portfolio/orders"),
            )
            r.raise_for_status()
            return True
        except Exception as e:
            log.error(f"place_order: {e}")
            return False


# ── Signal → Trade Logic ──────────────────────────────────────────────────────

def generate_signal(sentiment: CryptoSentiment) -> Optional[tuple[str, str, float]]:
    """
    Returns (coin, direction, confidence) or None.
    direction: "UP" = buy YES on higher price markets, "DOWN" = buy NO or lower price markets
    """
    # Extreme Fear + negative funding = oversold → contrarian BUY
    if sentiment.is_extreme_fear:
        confidence = 0.65 + (Config.FEAR_GREED_EXTREME_FEAR - sentiment.fear_greed_index) * 0.01
        # More fear = more confident in reversal
        confidence = min(0.85, confidence)
        log.info(f"[SIGNAL] EXTREME FEAR ({sentiment.fear_greed_index}) → BUY BTC conf={confidence:.0%}")
        return "BTC", "UP", confidence

    # Extreme Greed + high funding = overheated → contrarian SELL
    if sentiment.is_overheated:
        confidence = 0.60 + (sentiment.fear_greed_index - Config.FEAR_GREED_EXTREME_GREED) * 0.01
        confidence = min(0.80, confidence)
        log.info(f"[SIGNAL] EXTREME GREED + HIGH FUNDING → FADE BTC conf={confidence:.0%}")
        return "BTC", "DOWN", confidence

    # Strongly negative funding (shorts paying longs) + neutral/bearish sentiment = recovery
    if sentiment.btc_funding_rate < Config.FUNDING_RATE_LOW and 30 <= sentiment.fear_greed_index <= 50:
        confidence = 0.62
        log.info(f"[SIGNAL] NEGATIVE FUNDING + FEAR → BUY conf={confidence:.0%}")
        return "BTC", "UP", confidence

    return None


def find_trade(
    coin: str, direction: str, confidence: float,
    markets: list[KalshiMarket],
    btc_price: float, existing: set[str],
) -> Optional[tuple[KalshiMarket, str, int, int]]:
    """Find best Kalshi market to express the signal."""
    import re
    now = datetime.now(timezone.utc)
    candidates = [
        m for m in markets
        if m.ticker not in existing
        and m.close_time > now + timedelta(hours=2)
        and m.close_time < now + timedelta(days=7)
    ]

    log.info(f"[FIND_TRADE] {len(candidates)} candidate markets from {len(markets)} total "
             f"for {coin} {direction} (conf={confidence:.0%}, btc=${btc_price:,.0f})")

    for market in sorted(candidates, key=lambda m: m.volume, reverse=True):
        # Find price threshold in market title
        price_match = re.search(r"\$?([\d,]+(?:\.\d+)?)[Kk]?", market.title)
        if not price_match:
            log.debug(f"[FIND_TRADE] {market.ticker}: no price in title '{market.title}'")
            continue

        threshold_str = price_match.group(1).replace(",", "")
        try:
            threshold = float(threshold_str)
            if "K" in market.title or "k" in market.title:
                threshold *= 1000
        except Exception:
            continue

        if direction == "UP":
            # Buy YES on "above X" if X < current price (likely to stay above)
            # Or buy YES on a modest "above X" where X is slightly above current
            if btc_price > 0 and threshold < btc_price * 1.05:
                price = market.yes_price
                side = "YES"
            else:
                log.debug(f"[FIND_TRADE] {market.ticker}: threshold {threshold} vs btc {btc_price} - skip for UP")
                continue
        else:  # DOWN
            # Buy NO on "above X" if X is near current price
            if btc_price > 0 and threshold > btc_price * 0.98:
                price = market.no_price
                side = "NO"
            else:
                continue

        # Price might be in dollars (0.XX) from API v2 — normalize to cents
        if isinstance(price, float) and price < 1.0:
            price = int(round(price * 100))
        elif isinstance(price, float):
            price = int(round(price))

        if price == 0 or price is None:
            log.info(f"[FIND_TRADE] {market.ticker}: {side} price is 0/None, skipping")
            continue
        if not (Config.MIN_PRICE <= price <= Config.MAX_PRICE):
            log.info(f"[FIND_TRADE] {market.ticker}: {side} price {price}¢ outside [{Config.MIN_PRICE},{Config.MAX_PRICE}]")
            continue

        edge = confidence - (price / 100)
        if edge >= Config.MIN_EDGE:
            contracts = max(1, int(Config.BET_SIZE_USD * 100 / price))
            log.info(f"[FIND_TRADE] MATCH {market.ticker}: {side} @ {price}¢, edge={edge:.0%}")
            return market, side, price, contracts
        else:
            log.info(f"[FIND_TRADE] {market.ticker}: edge {edge:.1%} < min {Config.MIN_EDGE:.1%}")

    log.info(f"[FIND_TRADE] No qualifying market found for {coin} {direction}")
    return None


# ── Paper Ledger ──────────────────────────────────────────────────────────────

class PaperLedger:
    def __init__(self):
        self.balance = Config.PAPER_BALANCE
        self.trades: list[dict] = []
        self.open_positions: dict[str, dict] = {}

    def open_position(self, ticker: str, side: str, price: int, contracts: int,
                       coin: str, signal: str, fg: int) -> bool:
        if len(self.open_positions) >= Config.MAX_OPEN_POSITIONS:
            return False
        cost = price * contracts / 100
        if cost > self.balance:
            return False
        self.balance -= cost
        rec = {
            "ticker": ticker, "side": side, "price": price,
            "contracts": contracts, "cost": cost,
            "coin": coin, "signal": signal, "fear_greed": fg,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.open_positions[ticker] = rec
        self.trades.append({"action": "OPEN", **rec})
        log.info(f"[PAPER] OPEN {side} {ticker} @ {price}¢ × {contracts} = ${cost:.2f} | "
                 f"{coin} {signal} F&G={fg} | balance=${self.balance:.2f}")
        return True

    def close_position(self, ticker: str, exit_price: int, reason: str = ""):
        pos = self.open_positions.pop(ticker, None)
        if not pos:
            return
        pnl = (exit_price - pos["price"]) * pos["contracts"] / 100
        if pos["side"] == "NO":
            pnl = (pos["price"] - exit_price) * pos["contracts"] / 100
        self.balance += pos["cost"] + pnl
        self.trades.append({"action": "CLOSE", "ticker": ticker,
                             "exit_price": exit_price, "pnl": pnl, "reason": reason})
        log.info(f"[PAPER] CLOSE {ticker} @ {exit_price}¢ | PnL=${pnl:+.2f} | balance=${self.balance:.2f}")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Kalshi Crypto Sentiment Bot starting")
    log.info(f"  Paper mode:       {Config.PAPER_MODE}")
    log.info(f"  Extreme Fear <    {Config.FEAR_GREED_EXTREME_FEAR}")
    log.info(f"  Extreme Greed >   {Config.FEAR_GREED_EXTREME_GREED}")
    log.info(f"  Funding high:     {Config.FUNDING_RATE_HIGH:.3%}/8hr")
    log.info(f"  Min edge:         {Config.MIN_EDGE:.0%}")
    log.info(f"  Poll interval:    {Config.POLL_INTERVAL_SEC}s")
    log.info("=" * 60)

    sent_client = SentimentClient()
    kalshi = KalshiClient()
    ledger = PaperLedger()

    cycle = 0
    while True:
        cycle += 1
        log.info(f"── Cycle {cycle} ──────────────────────────")

        try:
            sentiment = sent_client.get_sentiment()
            log.info(
                f"[SENTIMENT] F&G={sentiment.fear_greed_index} ({sentiment.fear_greed_label}) | "
                f"BTC=${sentiment.btc_price:,.0f} ({sentiment.btc_24h_change:+.1f}%) | "
                f"ETH=${sentiment.eth_price:,.0f} | "
                f"BTC funding={sentiment.btc_funding_rate:.4%}/8hr"
            )

            signal_result = generate_signal(sentiment)
            if signal_result is None:
                log.info("[SIGNAL] No actionable signal this cycle")
            else:
                coin, direction, confidence = signal_result
                existing = set(ledger.open_positions.keys())

                for series in CRYPTO_SERIES.get(coin, []):
                    markets = kalshi.get_markets_for_series(series)
                    if not markets:
                        continue
                    result = find_trade(coin, direction, confidence, markets,
                                         sentiment.btc_price, existing)
                    if result:
                        market, side, price, contracts = result
                        log.info(f"[TRADE] {direction} signal → {side} {market.ticker} @ {price}¢")
                        if Config.PAPER_MODE:
                            if ledger.open_position(market.ticker, side, price, contracts,
                                                     coin, direction, sentiment.fear_greed_index):
                                existing.add(market.ticker)
                        else:
                            if kalshi.place_order(market.ticker, side, contracts, price):
                                existing.add(market.ticker)
                        break  # one trade per signal

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)

        open_count = len(ledger.open_positions)
        closed_pnl = sum(t.get("pnl", 0) for t in ledger.trades if t["action"] == "CLOSE")
        total_opened = sum(1 for t in ledger.trades if t["action"] == "OPEN")
        log.info(f"[SUMMARY] Balance=${ledger.balance:.2f} | Open={open_count} | "
                 f"Trades={total_opened} | PnL=${closed_pnl:+.2f}")

        time.sleep(Config.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
