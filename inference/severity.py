"""Rule-based severity estimate. This is an AI estimate, NOT an insurance or mechanical assessment.
Rules are heuristics keyed on class-name keywords; tune them for your dataset's classes."""
LEVELS = ["Low", "Medium", "High"]
HIGH_KW = ("shatter", "broken", "flat", "missing")
MED_KW = ("dent", "crack")


def base_level(name: str) -> int:
    n = name.lower()
    if any(k in n for k in HIGH_KW): return 2
    if any(k in n for k in MED_KW): return 1
    return 0


def detection_severity(name: str, area_pct: float) -> str:
    lvl = base_level(name)
    if area_pct >= 5: lvl += 1          # large relative to the image
    return LEVELS[min(lvl, 2)]


def overall_severity(dets: list, total_area_pct: float) -> str:
    if not dets: return "None"
    lvl = max(LEVELS.index(d["severity"]) for d in dets)
    if total_area_pct >= 15 or len(dets) >= 5: lvl = 2
    elif total_area_pct >= 5 or len(dets) >= 3: lvl = max(lvl, 1)
    return LEVELS[lvl]
