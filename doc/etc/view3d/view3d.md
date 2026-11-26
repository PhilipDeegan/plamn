# view3d

`etc/view3d/view3d.cpp`: a live 3D view of a running simulation (`python3 -m plamn <input>
--view3d`: plamn's watcher writes frames to the viewer's stdin). Each frame's points (cell
centres, every level, finer ones smaller) are coloured by value, the extremes opaque and the
middle of the range faint, so the inside of a 3D cloud shows.

Built on sebsjames/mathplot (C++20 modules, clang >= 20 only) at `tpp/mathplot`, with
`python3 res/build_3d_view.py` from the plamn root. `etc/view3d/fake_frames.py` writes synthetic
frames to try it without a simulation.

## Frames

Little endian, repeated until EOF; the latest one is drawn:

| type | field |
|------|-------|
| `char[4]` | `"PLMF"` |
| `u32` | title length, then the title's bytes |
| `f64` | t |
| `u64` | step |
| `u64` | n points |
| `f32[3n]` | x y z per point |
| `f32[n]` | value per point |
| `f32[n]` | radius per point (half its level's cell width) |

and optionally, after it, the frame's arrows (a vector field averaged over blocks of cells):

| type | field |
|------|-------|
| `char[4]` | `"PLMV"` |
| `u64` | n arrows |
| `f32[3n]` | x y z per arrow (its block's centre) |
| `f32[3n]` | the vector |
| `f32[n]` | half the block's width (slices: an arrow is in a plane its block crosses) |

After EOF the last frame stays until the window is closed.

## Keys

- moving around: the mouse as mathplot's (primary drag rotates, secondary drag translates, the
  wheel zooms; Ctrl-h prints its keys), and first person keys: W/S forward / back, A/D left /
  right, R/F up / down (a step: 2% of the data's extent; hold Shift for 10x), the arrow keys turn
  the view (3 degrees), Q/E roll. Ctrl-a resets the view.
- slices, one cell thick at each point's level (lo <= x - pos < hi: a level's cell centres hit
  once): 1/2/3 the plane normal to x/y/z, 4/5/6 two of them (xy, xz, yz), 7 all three, 8/9 move
  them down / up a finest cell, 0 the whole cloud (and the planes back to the middle). Slices
  draw every point in them (the instance cap only bites the whole cloud).
- arrows: mathplot's QuiverVisual over the balls, length by magnitude (the longest a block's
  width), colour by magnitude; filtered by the slices too.
- B: the balls shown / hidden, V: the arrows shown / hidden (kept over new frames).
- P: play / pause, N: one more frame while paused. With `--control-fd N` (the watcher's pipe) the
  viewer writes `pause` / `play` / `step` lines there and the simulation waits while paused;
  without one, pausing just keeps the current frame.
