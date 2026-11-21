#!/usr/bin/env bash
# what CI runs (build_nix.yml, format.yml), locally: before marking a PR ready, instead of
# spending Actions minutes. From the repository root; JOBS=8 compile jobs by default;
# PYPHARE: the PHARE checkout pyphare comes from (default /mkn/git/phare/wrangler: CI's).
#   res/ci/local.sh [base, default origin/next: PRs target next]
set -euo pipefail
BASE=${1:-origin/master} # as CI: the PR base (next -> master)
J=${JOBS:-8}
PYPHARE=${PYPHARE:-/mkn/git/phare/wrangler}
mb=$(git merge-base "$BASE" HEAD)

step() { echo; echo "== $*"; }

# warnings as CI's: -W 9 (-Wall -Wextra -pedantic -Werror with gcc), PHARE's devMode exceptions
#  (unused parameters, variables), and strict aliasing a warning only (nanobind's sources)
W=(-W 9 -a "-Wno-unused-parameter -Wno-unused-variable -Wno-error=strict-aliasing")

step "core tests"
mkn build test -p test_plamn_core -dKO -t "$J" "${W[@]}" -x res/mkn/thrust

step "slow tests compile (CI's slow.yml runs them)"
mkn build -p test_plamn_core_slow -dKO -t "$J" "${W[@]}" -x res/mkn/mpi

step "python modules (1D, 2D; 1D MHD only, ideal and Hall)"
./mkn.py --dim 1 2 -- "${W[@]}"
./mkn.py --mhd --no-etc --hall false true -- "${W[@]}"
./mkn.py --coupled --no-etc --dim 1 -- "${W[@]}" # MHD under hybrid (test_coupled)

step "C++ diagnostics tests (embedded python): 1 rank (mkn test), then 3"
mkn build test -p test_plamn_diagnostics -dKO -t "$J" "${W[@]}" -x res/mkn/mpi
for t in bin/test_plamn_diagnostics/test/*_test_diagnostics_*d.cpp; do
  mpirun --oversubscribe -n 3 "$t"
done

step "python tests: one process each, the 1 and 3 rank cases at the same time (as CI)"
PYTHONPATH=.:$PYPHARE/pyphare:tst/plamn/python \
  python3 -m phlop.run.test_cases -i tst -c 4 --tags mpi

step "clang-format (changed lines)"
# (git clang-format --diff exits 1 when it has a diff: under pipefail a pipe into grep never
#  failed this step; capture the diff, then look at it)
fmt=$(git clang-format-22 --binary clang-format-22 --extensions c,cc,cpp,h,hpp,ipp --diff "$mb" -- inc src tst || true)
if grep -q "^diff" <<<"$fmt"; then
  echo "$fmt"
  echo "clang-format would change files: git clang-format-22 $mb"
  exit 1
fi

step "iwyu (applying its fixes changes nothing)"
mkn build -dKOqp iwyu_fix -t "$J" -Wx res/mkn/mpi >/dev/null
git diff --exit-code -- inc src tst

step "clang-tidy (changed lines)"
python3 res/ci/compile_db.py test_plamn_core,py_etc,py_sim -dKO -x res/mkn/mpi
git diff -U0 "$mb" -- inc src tst \
  | clang-tidy-diff-22.py -clang-tidy-binary clang-tidy-22 -p1 -path . -j "$J" -quiet \
                          -iregex '.*\.(c|cc|cpp|h|hpp|ipp)' || true

echo; echo "== all done"
