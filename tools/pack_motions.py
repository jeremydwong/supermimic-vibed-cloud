"""Pack a directory of G1 retarget CSVs (LAFAN1 or BONES-SEED) into one 50 Hz .npz for MotionLib.from_packed.

    python tools/pack_motions.py data/bones_seed/g1/csv data/bones_seed/g1_50hz.npz \
        [--metadata data/bones_seed/metadata/seed_metadata_v004.parquet --drop_support_props --originals_only]

Parsing ~142k CSVs takes a while, so it runs once (in parallel) rather than at every training start.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supermimic.motion_io import read_clip, resample  # noqa: E402

MIN_FRAMES = 2  # finite differences need at least two frames per clip
# BONES-SEED props the body rests on, climbs or pushes against; the simulation has only a flat floor. Our reconstruction
# of the paper's filter: it keeps 130,304 of 142,220 motions (the paper reports 129,514). Hand-held objects are kept.
SUPPORT_PROPS = re.compile(r"box|crate|chair|wall|table|crutch|obstacle|ladder|stairs|pad|edge|bench|railing|^scaffolding$|tree \(scaffolding\)|log|cupboard|fridge")


def load(path):
    try:
        p, quat, j = resample(*read_clip(path))
    except Exception as e:  # a malformed file is reported and skipped, not fatal
        return path, None, repr(e)
    return path, (p.numpy(), quat.numpy(), j.numpy()), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory searched recursively for *.csv")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--metadata", help="BONES-SEED metadata parquet, needed by the filters below")
    ap.add_argument("--originals_only", action="store_true", help="drop the mirrored BONES-SEED copies")
    ap.add_argument("--drop_support_props", action="store_true", help="drop motions that need a box, chair, wall, ... (SUPPORT_PROPS)")
    ap.add_argument("--limit", type=int, default=0, help="pack only the first N files (for testing)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "**", "*.csv"), recursive=True))
    if args.originals_only or args.drop_support_props:
        import pandas as pd

        meta = pd.read_parquet(args.metadata, columns=["move_g1_path", "is_mirror", "content_props"])
        ok = pd.Series(True, index=meta.index)
        if args.originals_only:
            ok &= ~meta.is_mirror.astype(bool)
        if args.drop_support_props:
            props = meta.content_props.astype(str).str.lower().str.replace(",", ";").str.split(";")
            ok &= ~props.apply(lambda ts: any(SUPPORT_PROPS.search(t.strip()) for t in ts))
        keep = {os.path.basename(x) for x in meta.move_g1_path[ok]}
        files = [f for f in files if os.path.basename(f) in keep]
    if args.limit:
        files = files[: args.limit]
    print(f"packing {len(files)} files with {args.workers} workers")

    names, P, Q, J, lens, bad = [], [], [], [], [], []
    with Pool(args.workers) as pool:
        for i, (path, clip, err) in enumerate(pool.imap(load, files, chunksize=64)):
            if clip is None or len(clip[0]) < MIN_FRAMES:
                bad.append((path, err or "too short"))
                continue
            names.append(os.path.splitext(os.path.basename(path))[0])
            P.append(clip[0]), Q.append(clip[1]), J.append(clip[2]), lens.append(len(clip[0]))
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    for path, err in bad[:20]:
        print("skipped", path, err)
    n = sum(lens)
    print(f"{len(lens)} clips, {len(bad)} skipped, {n} frames = {n * 0.02 / 3600:.1f} h at 50 Hz")
    np.savez(args.out, names=np.array(names), lens=np.array(lens, dtype=np.int64),
             p=np.concatenate(P), quat=np.concatenate(Q), j=np.concatenate(J))  # fmt: skip
    print("wrote", args.out, f"{os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
