import cv2
import numpy as np
import pandas as pd

# ================================================================
# 0) UTILITAIRES
# ================================================================
def crop_roi_from_percent(img, x_pct, y_pct, w_pct, h_pct):
    H, W = img.shape[:2]
    x = int(x_pct * W)
    y = int(y_pct * H)
    w = int(w_pct * W)
    h = int(h_pct * H)
    return img[y:y+h, x:x+w].copy(), (x, y, w, h)


def group_positions(pos, min_gap=4):
    if len(pos) == 0:
        return []
    groups = [[pos[0]]]
    for p in pos[1:]:
        if p - groups[-1][-1] <= min_gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(np.mean(g)) for g in groups]


# ================================================================
# 1) ANALYSE FLOUE DES CELLULES
# ================================================================
def analyze_cells_fuzzy(cells, threshold_factor=1.8):
    black_sums = []

    for cell in cells:
        h = cell.shape[0]
        bottom = cell[h//2:, :]
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )
        black_sums.append(np.sum(th == 255))

    black_sums = np.array(black_sums)

    if len(black_sums) == 0:
        return None, "no_cells", 0.0

    reference = np.mean(black_sums)
    diffs = black_sums - reference
    max_diff = np.max(diffs)

    if max_diff <= 0:
        return None, "no_mark", 0.0

    # 🔷 FONCTION D’APPARTENANCE FLOUE
    mu = diffs / max_diff
    mu[mu < 0] = 0.0

    threshold = max_diff / threshold_factor
    candidates = np.where(diffs >= threshold)[0]

    if len(candidates) == 0:
        return None, "no_candidate", 0.0

    if len(candidates) == 1:
        idx = int(candidates[0])
        return idx, "single_candidate", float(mu[idx])

    if len(candidates) == 2:
        best, second = sorted(candidates, key=lambda i: black_sums[i], reverse=True)
        gap = (black_sums[best] - black_sums[second]) / max(black_sums[best], 1)
        if gap < 0.1:
            return None, "ambiguous_two", 0.0
        confidence = 0.6 * mu[best] + 0.4 * gap
        return int(best), "two_candidates_resolved", float(confidence)

    return None, "too_many_candidates", 0.0


# ================================================================
# 2) ANALYSE D’UNE PAGE
# ================================================================
def analyze_page(image_path):
    res = {
        "filename": image_path,
        "valid_grid": False,
        "status_int": "",
        "status_dec": "",
        "idx_int": np.nan,
        "idx_dec": np.nan,
        "note_detected": np.nan,
        "confidence_int": 0.0,
        "confidence_dec": 0.0,
        "confidence_global": 0.0,
        "error": ""
    }

    img = cv2.imread(image_path)
    if img is None:
        res["error"] = "image_not_found"
        return res

    # ---------- ROI ----------
    roi, _ = crop_roi_from_percent(img, 0.0, 0.2391, 0.98, 0.1056)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)

    # ---------- PROJECTIONS ----------
    v_proj = np.sum(th, axis=0)
    h_proj = np.sum(th, axis=1)

    v_lines = group_positions(np.where(v_proj > 0.7 * v_proj.max())[0], 6)
    h_lines = group_positions(np.where(h_proj > 0.7 * h_proj.max())[0], 6)

    if len(h_lines) < 2 or len(v_lines) < 5:
        res["error"] = "invalid_grid"
        return res

    y1, y2 = h_lines[0], h_lines[-1]
    diffs = np.diff(v_lines)
    split = int(np.argmax(diffs))

    cells_int, cells_dec = [], []

    for i in range(split):
        cells_int.append(roi[y1:y2, v_lines[i]:v_lines[i+1]])

    for i in range(split+1, len(v_lines)-1):
        cells_dec.append(roi[y1:y2, v_lines[i]:v_lines[i+1]])

    res["valid_grid"] = True

    # ---------- LOGIQUE FLOUE ----------
    idx_i, stat_i, conf_i = analyze_cells_fuzzy(cells_int)
    idx_d, stat_d, conf_d = analyze_cells_fuzzy(cells_dec)

    res["status_int"] = stat_i
    res["status_dec"] = stat_d
    res["confidence_int"] = conf_i
    res["confidence_dec"] = conf_d

    # ---------- VALIDATION DES DOMAINES ----------
    DEC_VALUES = [0.00, 0.25, 0.50, 0.75]

    if (
        idx_i is not None
        and idx_d is not None
        and 0 <= idx_d < len(DEC_VALUES)
    ):
        res["idx_int"] = idx_i
        res["idx_dec"] = idx_d
        res["note_detected"] = idx_i + DEC_VALUES[idx_d]
        res["confidence_global"] = 0.5 * conf_i + 0.5 * conf_d
    else:
        res["error"] = "no_reliable_note"

    return res


# ================================================================
# 3) BATCH + EXCEL
# ================================================================
results = []

for i in range(1, 71):
    fname = f"anis121125_page-{i:04d}.jpg"
    results.append(analyze_page(fname))

df = pd.DataFrame(results)
df.to_excel("resultats_anisk.xlsx", index=False)

print("Excel généré sans erreur, avec logique floue explicite et score de confiance.")
