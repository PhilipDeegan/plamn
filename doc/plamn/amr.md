# Hierarchies without data

`plamn/amr.py` chops and partitions a simulation's levels from python, before (and without)
running it, to compare decompositions.

```python
tag_boxes = {0: [domain], 1: [...], ...}  # e.g. pyphare's Tagger(sim).tag()
h = Chopper(sim, ranks=6, ranks_per_node=6, mode="even").chop(tag_boxes, weights)
h.levels[1]     # [(lower, upper, rank)]
h.metrics()     # per level: imbalance, cells exchanged
h.hierarchy()   # pharesee PatchHierarchy, "rank" field
```

## Chopping

Each level's boxes are cut into patches as the simulator does (`largest_patch_size`; refined
levels in whole coarser cells):

| Mode | Cut |
|---|---|
| `"even"` (the simulator's) | the patch count nearest the size, cells shared evenly, up to 25% past it |
| `"chop"` | from the lower corner, at most the size |

Each level is then partitioned along the Hilbert curve by patch weight, each node getting an
equal share, split over its ranks.

A patch weighs its cells times the particles per cell. On level 0, given a work map per level 0
cell (`Tagger.weights()`: the work above it too), it weighs its cells' sum of the map, so level 0
is cut by the work the levels above will bring.

## Metrics

`metrics()` (`damn::amr::metrics`) gives, per level:

- the patches, cells, weight per rank and imbalance (max over mean);
- the cells exchanged, by distance: same level ghosts within reach, and for levels above 0 the
  coarser cells under each patch grown by reach (what refining its ghosts reads).

| Distance | What |
|---|---|
| Self | own periodic image |
| Local | same rank |
| Numa | another memory domain of the same node |
| Node | another rank on the same node |
| Remote | another node |
