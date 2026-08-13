#!/bin/bash
# Put every pip-installed NVIDIA lib dir on the loader path, then run the given command
# in the qc env. GPU4PySCF resolves libcurand/libcublas/etc. through these.
NVROOT="$HOME/micromamba/envs/qc/lib/python3.11/site-packages/nvidia"
EXTRA=""
for d in "$NVROOT"/*/lib; do
  [ -d "$d" ] && EXTRA="$EXTRA:$d"
done
export LD_LIBRARY_PATH="${EXTRA#:}:$LD_LIBRARY_PATH"
exec "$HOME/.local/bin/micromamba" run -r "$HOME/micromamba" -n qc "$@"
