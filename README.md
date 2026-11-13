# plamn

Plasma physics on top of [damn](https://github.com/PhilipDeegan/damn), in the role PHARE plays on top of SAMRAI: quantities, grid layouts, fields, particles, ions, electrons, electromagnetics and their numerics.

damn is found at `tpp/damn`, cloned by mkn by default; for local development it can be a symlink to a damn checkout. damn names are always written in full (`damn::core::Box`).

```sh
mkn clean build test -p test_plamn_core -dKO -t 8
```
