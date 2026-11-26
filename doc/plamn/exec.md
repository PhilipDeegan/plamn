# Exec files

`plamn/exec.py`: where a run's MPI ranks go, from an exec file (e.g. `.plamn.exec.yml`, gitignored).

```yaml
hosts:
  - host: fad                # a hostname or an address
    ranks: 8                 # how many ranks this host takes
  - host: leg
    address: 10.20.113.192   # used when host does not resolve
    ranks: 4
    wrap: python3 /mkn/thermal.py --high 85 --low 75 --   # before each rank's command here
mpirun: --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0
bind: core                   # core, none, or an mpirun --map-by spec
```

- One host: `mpirun -n <ranks>` on it.
- More: one `mpirun` over all of them (`-H host:ranks,...`); the launching machine is given as
  `localhost` (by its LAN address prterun counts it as a second node and finds no slots).
- Over hosts, or with a `wrap`, each rank starts through `exec.py rank <exec file> <cwd> --
  <command>`: it finds its own host's entry (by hostname or one of the machine's addresses), makes
  and enters the run directory (it may not exist on the other hosts) and runs the command under
  the host's wrap.
- Open MPI's per host app contexts (`mpirun -H a ... : -H b ...`) are not used: prterun 5.0.7
  crashes on them.

`exec.py` has no plamn imports: scripts load it by path and ranks run it as a script, so it works
for any code's ranks, not only plamn's. The paths are the same on every host.
