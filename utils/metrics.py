import numpy as np

def infection_percentage(defect_mask, leaf_mask=None, bbox=None):
    """Compute infected area percentage."""
    if leaf_mask is not None:
        total = leaf_mask.sum()
        infected = np.logical_and(defect_mask, leaf_mask).sum()
        return (infected / total) * 100 if total > 0 else 0.0
    if bbox is not None:
        x, y, w, h = bbox
        total = w * h
        infected = defect_mask[y:y+h, x:x+w].sum()
        return (infected / total) * 100 if total > 0 else 0.0
    return (defect_mask.sum() / defect_mask.size) * 100

def spray_level_from_pct(pct):
    if pct < 5:
        return "Healthy (No Spray)"
    elif pct < 20:
        return "Low Spray"
    elif pct < 50:
        return "Medium Spray"
    else:
        return "High Spray"
