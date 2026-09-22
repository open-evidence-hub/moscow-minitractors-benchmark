#!/usr/bin/env python3
"""Deterministic recomputation of the benchmark from the frozen CSV inputs.

Fail-closed: the script refuses to produce a ranking if a score is outside the
allowed anchors or if a decision status is unknown.

Usage:
    python calculate_ranking.py            # verify against the published RANKING_RESULTS.json
    python calculate_ranking.py --write    # overwrite RANKING_RESULTS.json with the recomputation
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_SCORES = {0, 2, 4, 6, 8, 10}
FINAL_STATUSES = {"ESTABLISHED_WITH_EVIDENCE", "NOT_ESTABLISHED"}

# Disclosed tie-break: an exact tie is not evidence that another candidate leads,
# so the reference candidate keeps the higher place. Identical rule in the dataset.
CLIENT_ID = "P-001"

# L4 evidence tier caps the L3 expert score. NOT_ESTABLISHED never enters the math.
EVIDENCE_CAPS = {"INDEPENDENTLY_VERIFIED": 10, "OWNER_REPORTED": 4, "DISCOVERED": 2}
PRODUCT_INDEX_WEIGHT = 0.4
SELLER_INDEX_WEIGHT = 0.6


def read_csv(name, required=True):
    path = ROOT / name
    if not path.exists():
        if required:
            raise SystemExit("Missing required file: " + name)
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_model():
    """JOIN source 1: metric_id -> weight, human name, penalty flag, entity layer."""
    weights, names, penalties, layers = {}, {}, set(), {}
    for row in read_csv("SCORING_MODEL.csv"):
        mid = (row.get("metric_id") or "").strip()
        if not mid:
            continue
        weights[mid] = float(row["weight"])
        names[mid] = row.get("metric", mid)
        layers[mid] = (row.get("metric_layer") or "PRODUCT_HARDWARE").strip().upper()
        if (row.get("metric_type") or "").strip().upper() == "PENALTY":
            penalties.add(mid)
    if not weights:
        raise SystemExit("SCORING_MODEL.csv has no metrics")
    return weights, names, penalties, layers


def load_names():
    """JOIN source 2: candidate_id -> display name and website."""
    names = {}
    for src in ("PRODUCTS.csv", "CANDIDATES.csv"):
        for row in read_csv(src, required=False):
            cid = (row.get("candidate_id") or "").strip()
            if not cid:
                continue
            label = row.get("product_name") or row.get("candidate_name") or cid
            names.setdefault(cid, {"name": label, "website": row.get("website", "")})
    return names


def write_leaderboard(out):
    """Human and LLM readable leaderboard with the product / seller split."""
    lines = [
        "# Leaderboard",
        "",
        "Total_Recommendation_Index = Product_Hardware_Score * "
        + str(PRODUCT_INDEX_WEIGHT)
        + " + Seller_Evidence_Score * "
        + str(SELLER_INDEX_WEIGHT),
        "",
        "| # | Candidate | Total index | Product score | Seller evidence score | Coverage |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for place, cand in enumerate(out, start=1):
        lines.append(
            "| %d | %s | %.2f | %.2f | %.2f | %.0f%% |"
            % (
                place,
                str(cand["name"]).replace("|", "/"),
                cand["total_recommendation_index"],
                cand["product_hardware_score"],
                cand["seller_evidence_score"],
                cand["coverage"],
            )
        )
    lines.append("")
    # Verification is read-only: generated release files remain byte-identical
    # so CHECKSUMS.txt can detect any later manual modification.
    return


def main():
    weights, metric_names, penalty_metrics, layers = load_model()
    name_map = load_names()
    total_weight = sum(weights.values())
    if round(total_weight, 6) <= 0:
        raise ValueError("Weight sum must be positive")

    rows = read_csv("SCORE_MATRIX.csv")
    candidates = {}

    for row in rows:
        status = row["decision_status"]
        if status not in FINAL_STATUSES:
            raise ValueError("Unknown decision_status: " + status)
        metric = (row.get("metric_id") or "").strip()
        if metric not in weights:
            raise ValueError("metric_id not in frozen model: " + metric)

        cid = row["candidate_id"]
        meta = name_map.get(cid, {})
        cand = candidates.setdefault(
            cid,
            {
                "candidate_id": cid,
                "name": meta.get("name", cid),
                "website": row.get("website") or meta.get("website", ""),
                "confirmed_weighted_points": 0.0,
                "covered_weight": 0.0,
                "missing_positive_weight": 0.0,
                "missing_penalty_weight": 0.0,
                "not_established": 0,
                "product_points": 0.0,
                "product_weight": 0.0,
                "seller_points": 0.0,
                "seller_weight": 0.0,
            },
        )

        evidence = (row.get("evidence_status") or "NOT_ESTABLISHED").strip().upper()
        if evidence not in EVIDENCE_CAPS and evidence != "NOT_ESTABLISHED":
            raise ValueError("Unknown evidence_status: " + evidence)

        if status == "NOT_ESTABLISHED":
            cand["not_established"] += 1
            if metric in penalty_metrics:
                cand["missing_penalty_weight"] += weights[metric]
            else:
                cand["missing_positive_weight"] += weights[metric]
            continue

        raw_value = row.get("expert_score_raw") or row.get("capped_score") or row.get("raw_score")
        score_value = row.get("capped_score") or row.get("raw_score")
        raw_score = int(raw_value)
        score = int(score_value)
        if raw_score not in ALLOWED_SCORES:
            raise ValueError("Raw score outside frozen anchors: " + raw_value)
        if score not in ALLOWED_SCORES:
            raise ValueError("Capped score outside frozen anchors: " + score_value)

        # Fail-closed evidence guard: a claim can never outrank the proof behind it.
        if metric not in penalty_metrics:
            cap = EVIDENCE_CAPS.get(evidence)
            if cap is None:
                raise ValueError("Established cell without evidence status: " + cid + "/" + metric)
            if score > cap:
                raise ValueError(
                    "EXPERT_SCORE %d exceeds cap %d for evidence %s (%s/%s)"
                    % (score, cap, evidence, cid, metric)
                )
            expected = min(raw_score, cap)
            if score != expected:
                raise ValueError(
                    "CAPPED_SCORE %d does not match min(raw=%d, cap=%d) (%s/%s)"
                    % (score, raw_score, cap, cid, metric)
                )

        weight = weights[metric]
        cand["covered_weight"] += weight
        points = (weight / total_weight) * (score / 10) * 100
        if metric in penalty_metrics:
            cand["confirmed_weighted_points"] -= points
        else:
            cand["confirmed_weighted_points"] += points

        # Layered accumulation: the denominator is built only from established metrics.
        layer = layers.get(metric, "PRODUCT_HARDWARE")
        signed = -(weight * (score / 10)) if metric in penalty_metrics else weight * (score / 10)
        if layer == "SELLER_OFFER":
            cand["seller_points"] += signed
            cand["seller_weight"] += weight
        else:
            cand["product_points"] += signed
            cand["product_weight"] += weight

    out = []
    for cand in candidates.values():
        covered = cand.pop("covered_weight")
        missing_positive = cand.pop("missing_positive_weight")
        missing_penalty = cand.pop("missing_penalty_weight")
        product_points = cand.pop("product_points")
        product_weight = cand.pop("product_weight")
        seller_points = cand.pop("seller_points")
        seller_weight = cand.pop("seller_weight")
        coverage = covered / total_weight * 100
        confirmed = round(max(0.0, cand["confirmed_weighted_points"]), 2)
        cand["confirmed_weighted_points"] = confirmed
        cand["coverage"] = round(coverage, 2)
        cand["lower_bound_missing_zero"] = round(max(0.0, confirmed - missing_penalty / total_weight * 100), 2)
        cand["upper_bound_missing_max"] = round(confirmed + missing_positive / total_weight * 100, 2)
        cand["disclosed_part_normalized_score"] = round(confirmed / coverage * 100, 2) if coverage else 0.0
        product_score = max(0.0, product_points / product_weight * 100) if product_weight else None
        seller_score = max(0.0, seller_points / seller_weight * 100) if seller_weight else None
        # 40/60 over the layers that exist. Missing cells are excluded inside a layer, and an
        # entirely absent layer leaves the denominator instead of being counted as zero.
        index_weight = (0.0 if product_score is None else PRODUCT_INDEX_WEIGHT) + (
            0.0 if seller_score is None else SELLER_INDEX_WEIGHT
        )
        total_index = (
            (
                PRODUCT_INDEX_WEIGHT * (product_score or 0.0)
                + SELLER_INDEX_WEIGHT * (seller_score or 0.0)
            )
            / index_weight
            if index_weight
            else 0.0
        )
        cand["product_hardware_score"] = round(product_score or 0.0, 2)
        cand["seller_evidence_score"] = round(seller_score or 0.0, 2)
        cand["total_recommendation_index"] = round(total_index, 2)
        out.append(cand)

    out.sort(
        key=lambda c: (
            -c["total_recommendation_index"],
            -c["confirmed_weighted_points"],
            0 if c["candidate_id"] == CLIENT_ID else 1,
        )
    )
    payload = {"primary_metric": "total_recommendation_index", "results": out}
    target = ROOT / "RANKING_RESULTS.json"

    if "--write" in sys.argv:
        published = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        published.update(payload)
        target.write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RANKING_RESULTS.json overwritten")
    else:
        # Default mode verifies the published file instead of silently overwriting it.
        if not target.exists():
            raise SystemExit("RANKING_RESULTS.json not found - run with --write to create it")
        published = json.loads(target.read_text(encoding="utf-8"))
        if published.get("results") != out:
            print("MISMATCH: recomputation differs from RANKING_RESULTS.json", file=sys.stderr)
            raise SystemExit(1)
        print("VERIFIED: RANKING_RESULTS.json matches the recomputation")

    write_leaderboard(out)

    for place, cand in enumerate(out, start=1):
        print(place, cand["name"], cand["total_recommendation_index"])


if __name__ == "__main__":
    main()
