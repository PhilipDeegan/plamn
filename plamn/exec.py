#!/usr/bin/env python3
"""Where a run's MPI ranks go, from an exec file: doc/plamn/exec.md"""

import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import yaml

ENV = "PLAMN_EXEC"


def load(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text()) or {}
    hosts = spec.get("hosts") or []
    if not hosts:
        raise ValueError(f"{path}: no hosts")
    for h in hosts:
        if "host" not in h or "ranks" not in h:
            raise ValueError(f"{path}: each host needs host and ranks: {h}")
    return dict(spec, hosts=hosts, path=path)


def resolvable(name):
    try:
        socket.getaddrinfo(name, None)
        return True
    except socket.gaierror:
        return False


def mpi_host(h, mine):
    """the name mpirun is given: localhost for the launching machine (by its LAN address prterun
    counts it as a second node and finds no slots), else host if it resolves, else address"""
    if {h["host"], h.get("address")} & mine:
        return "localhost"
    if resolvable(h["host"]) or "address" not in h:
        return h["host"]
    return h["address"]


def local_addresses():
    names = {socket.gethostname(), "localhost", "127.0.0.1"}
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            names.update(a.address for a in addrs)
    except ImportError:
        names.update(
            subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        )
    return names


def this_host(spec):
    mine = local_addresses()
    for h in spec["hosts"]:
        if {h["host"], h.get("address")} & mine:
            return h
    return None


def total_ranks(spec):
    return sum(int(h["ranks"]) for h in spec["hosts"])


def mpirun(spec, cwd, env_names=(), np=None, bind=None):
    """the command before a rank's own: mpirun, its options, and, over hosts, this file as rank"""
    np = np or total_ranks(spec)
    bind = bind or spec.get("bind", "core")
    cmd = ["mpirun", "-n", str(np)]
    cmd += {"core": ["--bind-to", "core"], "none": ["--bind-to", "none"]}.get(
        bind, ["--map-by", bind, "--bind-to", "core"]
    )
    if len(spec["hosts"]) > 1:
        mine = local_addresses()
        cmd += ["-H", ",".join(f"{mpi_host(h, mine)}:{h['ranks']}" for h in spec["hosts"])]
        cmd += ["--wdir", str(Path.home())]  # the run directory is made by each rank, see rank()
        for name in env_names:
            cmd += ["-x", name]
    cmd += shlex.split(spec.get("mpirun", ""))
    if len(spec["hosts"]) == 1 and "wrap" not in spec["hosts"][0]:
        return cmd
    return cmd + [
        sys.executable,
        str(Path(__file__).resolve()),
        "rank",
        str(spec["path"]),
        str(cwd),
        "--",
    ]


def rank(argv):
    """python3 exec.py rank <exec file> <cwd> -- <command>: one rank, on whichever host it landed"""
    spec_path, cwd, sep, *cmd = argv
    if sep != "--" or not cmd:
        raise SystemExit("usage: exec.py rank <exec file> <cwd> -- <command>")
    Path(cwd).mkdir(parents=True, exist_ok=True)
    os.chdir(cwd)
    host = this_host(load(spec_path))
    wrap = shlex.split(host.get("wrap", "")) if host else []
    os.execvp((wrap + cmd)[0], wrap + cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rank":
        rank(sys.argv[2:])
    else:
        spec = load(sys.argv[1] if len(sys.argv) > 1 else os.environ[ENV])
        print(shlex.join(mpirun(spec, Path.cwd())), "<command>")
