from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import xml.etree.ElementTree as ET

import requests


@dataclass(slots=True)
class HeadlineBrief:
    headlines: list[dict]
    summary: str
    actions: list[dict]
    major_events: list[dict]
    generated_at: str


class MacroHeadlineBriefingEngine:
    """
    Pulls latest US/EU market-impact headlines and creates an actionable brief.
    """

    FEEDS = [
        ("us_markets", "https://news.google.com/rss/search?q=US+stocks+markets+finance&hl=en-US&gl=US&ceid=US:en"),
        ("europe_markets", "https://news.google.com/rss/search?q=Europe+stocks+markets+finance&hl=en-GB&gl=GB&ceid=GB:en"),
        ("world_macro", "https://news.google.com/rss/search?q=world+news+geopolitics+central+bank+elections+trade+sanctions+energy+shipping+health+disaster&hl=en-US&gl=US&ceid=US:en"),
        ("major_events", "https://news.google.com/rss/search?q=war+peace+treaty+earthquake+hurricane+flood+sanctions+pipeline+attack&hl=en-US&gl=US&ceid=US:en"),
        ("world_general", "https://news.google.com/rss/search?q=global+breaking+news+conflict+diplomacy+natural+disaster+outbreak+infrastructure+cyberattack&hl=en-US&gl=US&ceid=US:en"),
    ]

    THEME_MAP = [
        ("defense", ["war", "missile", "conflict", "military", "defense"], ["LMT", "NOC", "RTX", "GD", "XAR"], "Monitor defense contractors for momentum continuation."),
        ("energy", ["oil", "gas", "opec", "supply", "shipping"], ["XLE", "USO", "XOM", "CVX"], "Watch energy complex for supply-driven repricing."),
        ("rates", ["inflation", "fed", "ecb", "rate", "yield"], ["TLT", "IEF", "KRE", "XLF"], "Review rates-sensitive sectors and duration exposure."),
        ("technology", ["ai", "chip", "semiconductor", "cloud"], ["NVDA", "AMD", "MSFT", "QQQ"], "Track growth/tech beta for narrative acceleration."),
        ("risk_off", ["recession", "downgrade", "crisis", "default"], ["GLD", "TLT", "XLU"], "Consider defensive tilt and hedge overlays."),
    ]
    MAJOR_EVENT_MAP = [
        ("war_conflict", ["war", "missile", "airstrike", "invasion", "conflict", "military"], "High", ["LMT", "NOC", "RTX", "GD", "XAR", "GLD", "USO"]),
        ("peace_treaty", ["peace treaty", "ceasefire", "truce"], "Medium", ["SPY", "QQQ", "EFA", "IEF"]),
        ("natural_disaster", ["earthquake", "hurricane", "flood", "wildfire", "tsunami"], "High", ["CAT", "URI", "XLE", "DBA", "XLU"]),
        ("sanctions_trade", ["sanction", "embargo", "trade ban", "tariff"], "High", ["USO", "GLD", "XLE", "LMT", "TLT"]),
        ("pandemic_health", ["outbreak", "pandemic", "epidemic"], "High", ["XLV", "MRNA", "PFE", "TLT"]),
        ("diplomatic_shift", ["summit", "diplomatic", "treaty", "alliance", "negotiation"], "Medium", ["EFA", "VEA", "SPY", "QQQ"]),
        ("critical_infra", ["pipeline", "port", "shipping", "canal", "power grid", "blackout"], "High", ["USO", "XLE", "XLI", "GLD"]),
        ("cyber_event", ["cyberattack", "ransomware", "data breach", "critical system"], "Medium", ["PANW", "CRWD", "ZS", "QQQ"]),
    ]

    def __init__(self, timeout: int = 8) -> None:
        self.timeout = timeout
        self.session = requests.Session()

    def _pull_feed(self, label: str, url: str) -> list[dict]:
        out: list[dict] = []
        try:
            resp = self.session.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:8]:
                title = (item.findtext("title") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title:
                    continue
                out.append({"region": label, "title": title, "published": pub, "link": link})
        except Exception:
            pass
        return out

    def _major_events(self, heads: list[dict]) -> list[dict]:
        out: list[dict] = []
        for h in heads:
            t = h["title"].lower()
            for event, keys, severity, watch in self.MAJOR_EVENT_MAP:
                if any(k in t for k in keys):
                    out.append(
                        {
                            "event_type": event,
                            "headline": h["title"],
                            "region": h["region"],
                            "severity": severity,
                            "watch": watch,
                        }
                    )
        # dedupe by headline
        seen = set()
        dedup = []
        for e in out:
            key = (e["event_type"], e["headline"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(e)
        return dedup[:8]

    def _summarize(self, heads: list[dict]) -> tuple[str, list[dict], list[dict]]:
        text = " ".join(h["title"] for h in heads).lower()
        actions: list[dict] = []
        active = []
        major = self._major_events(heads)
        for theme, keys, tickers, action in self.THEME_MAP:
            if any(k in text for k in keys):
                active.append(theme)
                actions.append({"theme": theme, "action": action, "watch": tickers})
        for e in major:
            actions.append(
                {
                    "theme": f"major_event:{e['event_type']}",
                    "action": f"Major event risk detected ({e['severity']}): monitor spillover and gap risk.",
                    "watch": e["watch"],
                }
            )
        if not active:
            summary = (
                "Headline flow is mixed with no single dominant macro theme. "
                "Keep a balanced stance, prioritize high-confidence setups, and avoid over-concentration."
            )
            actions.append({"theme": "balanced", "action": "Maintain balanced exposure and monitor catalyst acceleration.", "watch": ["SPY", "QQQ", "TLT", "GLD"]})
            return summary, actions, major

        summary = (
            f"Current headline tone is led by {', '.join(active)} narratives. "
            "These themes can rotate capital quickly across sectors, so align tactical exposure with the active narrative while maintaining risk controls."
        )
        return summary, actions, major

    def generate(self) -> HeadlineBrief:
        heads: list[dict] = []
        for label, url in self.FEEDS:
            heads.extend(self._pull_feed(label, url))
        # keep latest unique titles
        seen = set()
        dedup = []
        for h in heads:
            t = h["title"]
            if t in seen:
                continue
            seen.add(t)
            dedup.append(h)
        dedup = dedup[:15]
        summary, actions, major_events = self._summarize(dedup)
        return HeadlineBrief(
            headlines=dedup,
            summary=summary,
            actions=actions + ([{"theme": "major_events", "action": "See major events panel for high-impact global catalysts.", "watch": []}] if major_events else []),
            major_events=major_events,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
