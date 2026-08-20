#!/usr/bin/env python3
"""
Find the identity pairs most at risk of being confused. Build plan 7.1,
answering PS-3 Q3.3a.

You cannot claim to have fixed a false match you never observed. Rather than
wait for a customer to report one, find the pair in YOUR OWN gallery with the
smallest separation: that is the John/David geometry, and it is your next
support ticket.
"""
import json, os, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, psycopg2

def main():
    cx = psycopg2.connect(os.environ["DATABASE_URL"]); cur = cx.cursor()
    cur.execute("SELECT subject_id, centroid FROM identities WHERE centroid IS NOT NULL")
    rows = cur.fetchall()
    if len(rows) < 2:
        sys.exit(f"need >=2 identities with centroids, have {len(rows)}")
    ids = [r[0] for r in rows]
    V = np.array([r[1] for r in rows], np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    pairs = []
    for i in range(len(ids)):
        j = int(np.argmax(S[i]))
        if j > i:
            pairs.append({"a": ids[i], "b": ids[j], "sim": round(float(S[i, j]), 4)})
    pairs.sort(key=lambda p: -p["sim"])
    out = {"n_identities": len(ids),
           "top_confusable": pairs[:20],
           "pairs_above_0.75": sum(1 for p in pairs if p["sim"] > 0.75),
           "pairs_above_0.85": sum(1 for p in pairs if p["sim"] > 0.85),
           "pairs_above_0.95": sum(1 for p in pairs if p["sim"] > 0.95),
           "max_sim": pairs[0]["sim"] if pairs else None,
           "interpretation": (
               "Each row is two DISTINCT identities whose centroids are nearly "
               "identical. At these similarities a single top-1 threshold "
               "cannot separate them; only the margin to the runner-up can, "
               "and where the margin is also small the correct output is "
               "REVIEW, not a confident answer.")}
    d = REPO / "results" / "privacy"; d.mkdir(parents=True, exist_ok=True)
    (d / "fp_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(f"  {len(ids)} identities, {len(pairs)} nearest-neighbour pairs")
    print(f"  above 0.75: {out['pairs_above_0.75']}   above 0.85: "
          f"{out['pairs_above_0.85']}   above 0.95: {out['pairs_above_0.95']}")
    for p in pairs[:5]:
        print(f"    {p['a']}  <->  {p['b']}   sim={p['sim']}")
    print(f"  wrote {d/'fp_diagnosis.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
