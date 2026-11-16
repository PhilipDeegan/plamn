#!/usr/bin/env bash
# what CI runs (build_nix.yml, format.yml), locally: before marking a PR ready, instead of
# spending Actions minutes. From the repository root; JOBS=8 compile jobs by default;
# PYPHARE: the PHARE checkout pyphare comes from (default ~/git/phare/mkn).
#   res/ci/local.sh [base, default origin/master]
set -euo pipefail
BASE=${1:-origin/master}
J=${JOBS:-8}
PYPHARE=${PYPHARE:-$HOME/git/phare/mkn}
mb=$(git merge-base "$BASE" HEAD)

step() { echo; echo "== $*"; }

step "core tests"
mkn build test -p test_plamn_core -dKO -t "$J"

step "python modules (1D, 2D)"
./mkn.py --dim 1 2

step "python tests: one process each, the 1 and 3 rank cases at the same time (as CI)"
PYTHONPATH=python:$PYPHARE/pyphare:tst/plamn/python \
  python3 -m phlop.run.test_cases -i tst -c 4 --tags mpi

step "clang-format (changed lines)"
git clang-format-22 --binary clang-format-22 --extensions c,cc,cpp,h,hpp,ipp --diff "$mb" -- inc src tst \
  | grep -q "^diff" && { echo "clang-format would change files: git clang-format-22 $mb"; exit 1; } || true

step "iwyu (applying its fixes changes nothing)"
mkn build -dKOqp iwyu_fix -t "$J" -Wx res/mkn/mpi >/dev/null
git diff --exit-code -- inc src tst

step "clang-tidy (changed lines)"
python3 res/ci/compile_db.py test_plamn_core,py_etc,py_sim -dKO -x res/mkn/mpi
git diff -U0 "$mb" -- inc src tst \
  | clang-tidy-diff-22.py -clang-tidy-binary clang-tidy-22 -p1 -path . -j "$J" -quiet \
                          -iregex '.*\.(c|cc|cpp|h|hpp|ipp)' || true

echo; echo "== all done"
