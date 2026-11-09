"""Pack G1 retarget CSVs (LAFAN1 or BONES-SEED) into one 50 Hz .npz for MotionLib.from_packed.

    python tools/pack_motions.py data/bones_seed/g1/csv data/bones_seed/g1_50hz.npz \
        [--metadata data/bones_seed/metadata/seed_metadata_v004.parquet --drop_support_props --originals_only]
    curl ... g1.tar.gz | python tools/pack_motions.py - data/bones_seed/g1_50hz_noprops.npz --metadata ...

The source is a directory searched recursively for *.csv, or a tar(.gz) archive of them (a path, or - for stdin).
An archive is read as a stream, so its ~142k small CSVs never touch the disk (slow on RunPod's network volumes).
Parsing takes a while, so it runs once (in parallel) rather than at every training start.
"""

import argparse
import glob
import itertools
import os
import re
import sys
import tarfile
import threading
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supermimic.motion_io import read_clip, resample  # noqa: E402

MIN_FRAMES = 2  # finite differences need at least two frames per clip
IN_FLIGHT = 4096  # archive members read ahead of the parsing workers (bounds memory)
# BONES-SEED props the body rests on, climbs or pushes against; the simulation has only a flat floor. Our reconstruction
# of the paper's filter: it keeps 130,304 of 142,220 motions (the paper reports 129,514). Hand-held objects are kept.
SUPPORT_PROPS = re.compile(r"box|crate|chair|wall|table|crutch|obstacle|ladder|stairs|pad|edge|bench|railing|^scaffolding$|tree \(scaffolding\)|log|cupboard|fridge")


def cpu_count():
    """CPUs this process may use: the cgroup quota (a RunPod pod sees all of the host's cores), else the affinity."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def init_worker():
    torch.set_num_threads(1)  # parallelism is across clips; per-clip torch ops are tiny


def load(item):
    path, data = item  # data: the CSV's bytes, or None to read it from path
    try:
        p, quat, j = resample(*read_clip(path if data is None else data))
    except Exception as e:  # a malformed file is reported and skipped, not fatal
        return path, None, repr(e)
    return path, (p.numpy(), quat.numpy(), j.numpy()), None


def archive_members(src, wanted, slots):
    """(path, bytes) of the wanted *.csv members of a tar(.gz) stream, at most IN_FLIGHT ahead of the consumer."""
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") if src == "-" else tarfile.open(src, mode="r|*") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".csv") and wanted(m.name):
                slots.acquire()
                yield m.name, tar.extractfile(m).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory searched recursively for *.csv, or a tar(.gz) of them (- for stdin)")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--metadata", help="BONES-SEED metadata parquet, needed by the filters below")
    ap.add_argument("--originals_only", action="store_true", help="drop the mirrored BONES-SEED copies")
    ap.add_argument("--drop_support_props", action="store_true", help="drop motions that need a box, chair, wall, ... (SUPPORT_PROPS)")
    ap.add_argument("--limit", type=int, default=0, help="pack only the first N files (for testing)")
    ap.add_argument("--workers", type=int, default=cpu_count())
    args = ap.parse_args()

    keep = None
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
    wanted = lambda f: keep is None or os.path.basename(f) in keep  # noqa: E731

    slots = threading.BoundedSemaphore(IN_FLIGHT)
    if os.path.isdir(args.src):
        files = [f for f in sorted(glob.glob(os.path.join(args.src, "**", "*.csv"), recursive=True)) if wanted(f)]
        if args.limit:
            files = files[: args.limit]
        items, total = ((f, None) for f in files), len(files)
        print(f"packing {total} files with {args.workers} workers")
    else:
        items = archive_members(args.src, wanted, slots)
        if args.limit:
            items = itertools.islice(items, args.limit)
        total = "?"
        print(f"packing the CSVs of {args.src} with {args.workers} workers")

    clips, bad = {}, []
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, (path, clip, err) in enumerate(pool.imap(load, items, chunksize=64)):
            if not os.path.isdir(args.src):
                slots.release()
            if clip is None or len(clip[0]) < MIN_FRAMES:
                bad.append((path, err or "too short"))
            else:
                clips[path] = clip
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{total}", flush=True)
    for path, err in bad[:20]:
        print("skipped", path, err)
    paths = sorted(clips)  # the same order whether read from a directory or from an archive
    names = [os.path.splitext(os.path.basename(f))[0] for f in paths]
    lens = [len(clips[f][0]) for f in paths]
    n = sum(lens)
    print(f"{len(lens)} clips, {len(bad)} skipped, {n} frames = {n * 0.02 / 3600:.1f} h at 50 Hz")
    np.savez(args.out, names=np.array(names), lens=np.array(lens, dtype=np.int64),
             p=np.concatenate([clips[f][0] for f in paths]), quat=np.concatenate([clips[f][1] for f in paths]),
             j=np.concatenate([clips[f][2] for f in paths]))  # fmt: skip
    print("wrote", args.out, f"{os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
