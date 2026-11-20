"""Every comparison run's peak memory in one table: the OUT.mem.json files (test_ladder.memory)
under the given directories, one row per code and case, against the case's estimate and tier
(cases.py), so the estimates and tiers can be set from measured numbers.

  python3 etc/compare/phare/memory_table.py DIR... [--json FILE] [--md FILE]

The markdown table goes to stdout too (slow.yml appends it to the job summary).
"""

import json
import sys
from pathlib import Path

GB = 1 << 30


def rows(dirs):
    """the OUT.mem.json rows under dirs (the glob leaves out the per rank OUT.mem.<rank>.json)"""
    found = []
    for d in dirs:
        for f in sorted(Path(d).rglob("*.mem.json")):
            row = json.loads(f.read_text())
            row["file"] = str(f)
            found.append(row)
    return sorted(found, key=lambda r: (r["case"], r["code"]))


def tier(case):
    """the case's tier from cases.py, "?" when the case is not (or no longer) there"""
    import cases

    return cases.tier(case) if case in cases.CASES else "?"


def markdown(table):
    lines = [
        "| case | code | ranks | peak GB | largest rank GB | estimate GB | peak / estimate | tier | s |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in table:
        est = r.get("estimate") or 0
        ratio = f"{r['peak_rss'] / est:.2f}" if est else "-"
        lines.append(
            f"| {r['case']} | {Path(r['code']).name} | {r['ranks']} | {r['peak_rss'] / GB:.2f} "
            f"| {r['peak_rss_rank'] / GB:.2f} | {est / GB:.2f} | {ratio} | {r['tier']} "
            f"| {r['seconds']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--md")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    table = rows(args.dirs)
    for r in table:
        r["tier"] = tier(r["case"])
    md = markdown(table)
    print(md, end="")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    if args.md:
        Path(args.md).write_text(md)
    if not table:
        print(f"no *.mem.json under {' '.join(args.dirs)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
