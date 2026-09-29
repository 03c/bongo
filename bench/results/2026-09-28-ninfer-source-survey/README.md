# NInfer source survey (BAS-64, R2)

Raw evidence for `docs/research/ninfer-architecture.md`.

| | |
| --- | --- |
| Upstream | `https://github.com/Neroued/ninfer` |
| Commit | `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d` ("docs: align linear guidance and refresh q4 performance report", 2026-09-26) |
| Method | shallow clone, `GIT_LFS_SKIP_SMUDGE=1`, 75 MB, read-only, into the run scratch dir |
| Weights downloaded | none |
| Vendored into bongo | none |

`op-inventory.txt` records the commit, the `include/ninfer/ops/` contract list, the `src/ops/`
family list, the per-format registered shape files, the attention/recurrent family trees, and
every `docs/` file present. It was generated with `ls`/`find` only.

Reproduce:

```sh
S="${PAPERCLIP_RUN_SCRATCH_DIR:-/tmp}/ninfer-survey"; mkdir -p "$S" && cd "$S"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Neroued/ninfer.git ninfer
cd ninfer && git rev-parse HEAD   # e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d
```
