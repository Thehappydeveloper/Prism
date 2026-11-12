from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class NetworkProfile:
    id: str
    description: str
    delay_ms: float
    jitter_ms: float
    loss_pct: float
    rate_mbit: float | None
    delay_distribution: str = "uniform"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


UNLIMITED_RATE_BPS = 10_000_000_000.0


NETWORK_PROFILES: Dict[str, NetworkProfile] = {
    "ideal": NetworkProfile("ideal", "Ideal network - no impairments", 0, 0, 0, None),
    "good_wifi": NetworkProfile("good_wifi", "Good WiFi connection", 30, 10, 0.1, 50),
    # Matched fixed-rate profiles for the paper FoV-aware bandwidth study.
    # They deliberately retain great_wifi's delay, jitter, loss, and uniform
    # delay distribution, so rate is the only network variable.
    "paper_wifi_71": NetworkProfile("paper_wifi_71", "Paper WiFi: 71 Mb/s fixed rate", 30, 10, 0.1, 71),
    "paper_wifi_85": NetworkProfile("paper_wifi_85", "Paper WiFi: 85 Mb/s fixed rate", 30, 10, 0.1, 85),
    "paper_wifi_118": NetworkProfile("paper_wifi_118", "Paper WiFi: 118 Mb/s fixed rate", 30, 10, 0.1, 118),
    "paper_wifi_55": NetworkProfile("paper_wifi_55", "Paper WiFi: 55 Mb/s fixed rate", 30, 10, 0.1, 55),
    "paper_wifi_74": NetworkProfile("paper_wifi_74", "Paper WiFi: 74 Mb/s fixed rate", 30, 10, 0.1, 74),
    "paper_wifi_112": NetworkProfile("paper_wifi_112", "Paper WiFi: 112 Mb/s fixed rate", 30, 10, 0.1, 112),
    "paper_wifi_150": NetworkProfile("paper_wifi_150", "Paper WiFi: 150 Mb/s fixed rate", 30, 10, 0.1, 150),
    "paper_wifi_188": NetworkProfile("paper_wifi_188", "Paper WiFi: 188 Mb/s fixed rate", 30, 10, 0.1, 188),
    "paper_wifi_225": NetworkProfile("paper_wifi_225", "Paper WiFi: 225 Mb/s fixed rate", 30, 10, 0.1, 225),
    "great_wifi": NetworkProfile("great_wifi", "Great WiFi connection", 30, 10, 0.1, 175),
    "moderate_wifi": NetworkProfile("moderate_wifi", "Moderate WiFi connection", 30, 10, 0.1, 125),
    "poor_cellular": NetworkProfile("poor_cellular", "Poor cellular signal / cell edge", 120, 30, 1.0, 5),
    "satellite": NetworkProfile("satellite", "Satellite connection - high latency", 600, 50, 0.5, 10),
    "congested": NetworkProfile("congested", "Heavy network congestion", 200, 50, 3.0, 1),
    "dsl_asymmetric": NetworkProfile("dsl_asymmetric", "DSL-like asymmetric connection", 40, 5, 0.05, 20),
    "low_latency": NetworkProfile("low_latency", "Low latency optimized", 5, 1, 0, 100),
    "lossy": NetworkProfile("lossy", "High packet loss for testing", 20, 5, 5.0, None),
    "jittery": NetworkProfile("jittery", "High jitter connection", 50, 40, 0.1, None, delay_distribution="normal"),
    "lte_typical": NetworkProfile("lte_typical", "Typical 4G LTE conditions", 50, 15, 0.2, 30),
    "edge": NetworkProfile("edge", "Edge computing - low latency", 2, 1, 0, 200),
    "transatlantic": NetworkProfile("transatlantic", "Transatlantic connection", 80, 10, 0.1, 100),
}


def get_network_profile(profile_id: str | None) -> NetworkProfile | None:
    if not profile_id or profile_id == "None":
        return None
    return NETWORK_PROFILES.get(str(profile_id))


def network_profile_payload() -> list[dict[str, Any]]:
    return [profile.to_dict() for profile in NETWORK_PROFILES.values()]
