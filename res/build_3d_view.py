#!/usr/bin/env python3
"""build etc/view3d (python3 -m plamn ... --view3d): a live 3D viewer on sebsjames/mathplot

    python3 res/build_3d_view.py [-j N] [-p view3d|hello]      (from the plamn root)

needs clang >= 20 with its clang-scan-deps (mathplot is C++20 modules: gcc can't build it),
GL, GLFW (from mkn.gl, fetched by mkn), FreeType (libfreetype-dev). mathplot goes to
tpp/mathplot: cloned there (with its submodules), or MATHPLOT=<a clone> links that one.

maiken compiles each file on its own, modules need the modules they import precompiled first:
  1. mkn's dry run (-x res/mkn/clang_modules -R, from etc/view3d) as a compilation database
  2. clang-scan-deps -format=p1689: each unit's provided and required modules
  3. only the modules the main file needs (transitively), in import order: each --precompile
     to bin/<profile>/pcm/<name>.pcm then to an object, -fprebuilt-module-path (independent
     ones in parallel)
  4. the other units needing only those (implementation units, plain sources), the main file,
     mkn's link line with every object
the viewer: etc/view3d/bin/view3d/plamn.view3d
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "etc" / "view3d"
MATHPLOT = ROOT / "tpp" / "mathplot"
REPO = "https://github.com/sebsjames/mathplot"
SCAN = os.environ.get("CLANG_SCAN_DEPS", "")


def clang_check():
    """clang >= 20, and the clang-scan-deps of its version"""
    global SCAN
    out = subprocess.run(["clang++", "--version"], capture_output=True, text=True,
                         check=False).stdout
    m = re.search(r"clang version (\d+)", out)
    if not m or int(m.group(1)) < 20:
        raise SystemExit(f"view3d needs clang >= 20 (mathplot's C++20 modules), clang++: {out!r}")
    if not SCAN:
        SCAN = shutil.which(f"clang-scan-deps-{m.group(1)}") or shutil.which("clang-scan-deps")
    if not SCAN:
        raise SystemExit(f"no clang-scan-deps(-{m.group(1)}): install clang-tools")


def mathplot():
    """tpp/mathplot: MATHPLOT's clone linked, or cloned (shallow, with submodules)"""
    if MATHPLOT.exists():
        return
    MATHPLOT.parent.mkdir(exist_ok=True)
    if os.environ.get("MATHPLOT"):
        MATHPLOT.symlink_to(Path(os.environ["MATHPLOT"]).resolve())
        return
    subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules",
                    "--shallow-submodules", REPO, str(MATHPLOT)], check=True)


def dry_run(profile):
    out = subprocess.run(
        ["mkn", "build", "-x", str(ROOT / "res" / "mkn" / "clang_modules"), "-R", "-p", profile],
        cwd=HERE, check=True, capture_output=True, text=True,
    ).stdout
    compiles, link = [], None
    for line in out.splitlines():
        words = shlex.split(line)
        if words and words[0] == "ccache":
            words = words[1:]
        if not words or not words[0].startswith("clang"):
            continue
        if "-c" in words:
            src = words[words.index("-c") + 1]
            obj = words[words.index("-o") + 1]
            compiles.append({"words": words, "file": src, "output": obj})
        elif "-o" in words:
            link = words
    if link is None:
        raise RuntimeError("no link line in the dry run")
    # a string define mkn's args can't carry (quotes lost): the fonts mathplot embeds
    for c in compiles:
        c["words"].insert(1, f'-DMPLOT_FONTS_DIR="{MATHPLOT.resolve()}/fonts"')
    return compiles, link


def scan(compiles, build):
    db = build / "compile_commands.json"
    db.write_text(json.dumps([
        {"directory": str(HERE), "arguments": c["words"], "file": c["file"],
         "output": c["output"]} for c in compiles]))
    r = subprocess.run([SCAN, "-format=p1689", "-compilation-database", str(db), "-j", "4"],
                       capture_output=True, text=True, check=False)  # errors: below
    # units that fail to scan (e.g. hdfdata.cppm without serial HDF5) only matter if needed:
    #  their modules are then missing ("no unit provides module ...")
    for line in r.stderr.splitlines():
        if "error:" in line:
            print("scan: " + line, flush=True)
    return json.loads(r.stdout)["rules"]


def pcm_name(logical):
    return logical.replace(":", "-") + ".pcm"  # partitions: clang's prebuilt file name


def run(cmd):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=False)
    if r.returncode:
        sys.stderr.write(" ".join(cmd) + "\n" + r.stderr)
        raise SystemExit(r.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-j", type=int, default=4, help="parallel module compiles")
    p.add_argument("-p", "--profile", default="view3d", help="view3d, or hello (mathplot's)")
    args = p.parse_args()
    clang_check()
    mathplot()

    build = HERE / "bin" / args.profile
    pcm = build / "pcm"
    pcm.mkdir(parents=True, exist_ok=True)
    compiles, link = dry_run(args.profile)
    by_output = {c["output"]: c for c in compiles}
    rules = scan(compiles, build)

    provides, units = {}, []  # logical name -> rule; every rule
    for r in rules:
        c = by_output[r["primary-output"]]
        r["compile"] = c
        r["needs"] = [q["logical-name"] for q in r.get("requires", [])]
        for m in r.get("provides", []):
            provides[m["logical-name"]] = r
        units.append(r)
    main_file = profile_main(args.profile)
    main_unit = next(r for r in units if Path(r["compile"]["file"]).name == main_file)

    # the modules the main file needs, transitively
    order, seen = [], set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        if name not in provides:
            raise RuntimeError(f"no unit provides module {name}")
        for dep in provides[name]["needs"]:
            visit(dep)
        order.append(name)

    for name in main_unit["needs"]:
        visit(name)
    print(f"{len(order)} modules of {len(provides)} needed", flush=True)

    prebuilt = f"-fprebuilt-module-path={pcm}"
    cxx = compiles[0]["words"][0]
    objects, done = [], set()

    def build_module(name):
        c = provides[name]["compile"]
        words = c["words"]
        base = [w for i, w in enumerate(words) if i > 0 and w != "-c" and w != c["file"]
                and not (i > 0 and words[i - 1] == "-o") and w != "-o"]
        out = pcm / pcm_name(name)
        run([cxx, *base, prebuilt, "--precompile", "-x", "c++-module", c["file"], "-o", str(out)])
        obj = str(build / "obj" / (pcm_name(name)[:-4] + ".o"))
        Path(obj).parent.mkdir(parents=True, exist_ok=True)
        run([cxx, "-fPIC", prebuilt, "-c", str(out), "-o", obj])
        return name, obj

    # in waves: every module whose imports are done
    pending = list(order)
    with cf.ThreadPoolExecutor(max_workers=args.j) as ex:
        while pending:
            ready = [m for m in pending if all(d in done for d in provides[m]["needs"])]
            for name, obj in ex.map(build_module, ready):
                done.add(name)
                objects.append(obj)
                print(f"  {name}", flush=True)
            pending = [m for m in pending if m not in done]

    # other units needing only built modules (implementation units, plain sources), the main
    extra = [r for r in units if not r.get("provides") and r is not main_unit
             and all(n in done for n in r["needs"])]
    for r in extra + [main_unit]:
        c = r["compile"]
        (HERE / c["output"]).parent.mkdir(parents=True, exist_ok=True)
        run([c["words"][0], prebuilt, *c["words"][1:]])
        objects.append(c["output"])

    # mkn's link line, with every object
    words = link
    o = words.index("-o")
    head = [w for w in words[: o + 2]]
    rest = [w for w in words[o + 2:] if not w.endswith(".o")]
    run([*head, *objects, *rest])
    print(f"linked {words[o + 1]}", flush=True)


def profile_main(profile):
    """the file name of the profile's main: (mkn.yaml, through its parents)"""
    import yaml

    profiles = {p["name"]: p for p in yaml.safe_load((HERE / "mkn.yaml").read_text())["profile"]}
    name = profile
    while name:
        if "main" in profiles[name]:
            return Path(profiles[name]["main"]).name
        name = profiles[name].get("parent")
    raise RuntimeError(f"profile {profile} has no main")


if __name__ == "__main__":
    main()
