"""CUDA allocations and unified memory page faults per scope: each event attributed to the
innermost NVTX range open (phlop's scopes with -DPHLOP_NVTX, mkn.gpu's, thrust's / CUB's).

Build a module with the scopes as NVTX ranges, profile, export, attribute:
    ./mkn.py --dim 3 --layout SoAVXChunked --allocator GPU_UNIFIED --no-etc \\
        --scope-flags 1=time 2=time 3=time -- -g 0 -x res/mkn/clang_cuda_mpi \\
        -a "-DPHLOP_NVTX" -l "-ldl"
    nsys profile -o run -t cuda,nvtx --cuda-um-cpu-page-faults=true \\
        --cuda-um-gpu-page-faults=true python3 <case.py>
    nsys stats --report nvtx_sum run.nsys-rep   # writes run.sqlite next to it
    python3 etc/bench/gpu/nvtx_attr.py run.sqlite

Runtime calls go to the innermost range of their own thread; page faults (no thread recorded) to
the innermost range of any thread open at their time.
"""

import bisect
import collections
import sqlite3
import sys

ALLOCATIONS = ("cudaMallocManaged", "cudaMalloc", "cudaFree")


def load_ranges(db, names):
    per_thread = collections.defaultdict(list)
    every = []
    query = (
        "select start, end, text, textId, globalTid from NVTX_EVENTS"
        " where end is not null and end > start"
    )
    for start, end, text, text_id, tid in db.execute(query):
        r = (start, end, text or names.get(text_id, "?"))
        per_thread[tid].append(r)
        every.append(r)
    for v in per_thread.values():
        v.sort()
    every.sort()
    return per_thread, every


def innermost(ranges, t, scan=400):
    """the shortest range of the sorted ranges containing t (a bounded scan back)"""
    best = None
    i = bisect.bisect_right(ranges, (t, float("inf"), "")) - 1
    for j in range(i, max(-1, i - scan), -1):
        start, end, name = ranges[j]
        if start <= t <= end and (best is None or end - start < best[1] - best[0]):
            best = (start, end, name)
    return best[2] if best else "(none)"


def show(title, counts, top=14):
    print(f"\n{title} (total {sum(counts.values())})")
    for name, count in counts.most_common(top):
        print(f"{count:8d}  {name}")


def main(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    per_thread, every = load_ranges(db, names)

    calls = {fn: collections.Counter() for fn in ALLOCATIONS}
    for start, tid, name_id in db.execute(
        "select start, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        fn = names.get(name_id, "?").split("_")[0]
        if fn in calls:
            calls[fn][innermost(per_thread.get(tid, []), start)] += 1
    for fn in ALLOCATIONS:
        show(fn, calls[fn])

    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    if "CUDA_UM_CPU_PAGE_FAULT_EVENTS" in tables:
        cpu = collections.Counter(
            innermost(every, t)
            for (t,) in db.execute("select start from CUDA_UM_CPU_PAGE_FAULT_EVENTS")
        )
        show("UM CPU page faults", cpu)
    if "CUDA_UM_GPU_PAGE_FAULT_EVENTS" in tables:
        gpu = collections.Counter()
        for t, n in db.execute(
            "select start, numberOfPageFaults from CUDA_UM_GPU_PAGE_FAULT_EVENTS"
        ):
            gpu[innermost(every, t)] += n
        show("UM GPU page faults", gpu)


if __name__ == "__main__":
    main(sys.argv[1])
