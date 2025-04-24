# utils.py

def implied_probability(odds):
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def edge_level(edge):
    if edge >= 0.1:
        return "High"
    elif edge >= 0.05:
        return "Moderate"
    elif edge >= 0.01:
        return "Low"
    else:
        return "Negligible"

def to_py(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return obj

def certainty_level(certainty):
    if certainty >= 0.40:      # P ≥ 0.90  or ≤ 0.10
        return "High"
    elif certainty >= 0.25:    # P ≥ 0.75  or ≤ 0.25
        return "Moderate"
    elif certainty >= 0.10:    # P ≥ 0.60  or ≤ 0.40
        return "Low"
    else:
        return "Negligible"
