#!/usr/bin/env python3
"""Bit-exactness check for the cand8 SM120/121 SF-layout torch port (LOCAL OVERLAY in
overlay0261/vllm/utils/deep_gemm.py) against the prod cand7 image's DeepGEMM a6b593d
C++ binding.

Runs ON the head node inside a docker image (usage below). Emits one JSON line per case to
stdout: {"shape", "stride", "md5"} — run on both images and diff the outputs; they
must be IDENTICAL.

  ssh "$CLUSTER_USER@$HEAD_HOST"   # or your head-node ssh target
  docker run --rm --gpus all -v "$HOME/verify-sf-layout.py:/tmp/v.py" \
      --entrypoint python3 <IMAGE> /tmp/v.py > out.<lane>.jsonl

Cases cover the DSv4 fp8 block-scale shapes' geometry classes: aligned/odd mn,
aligned/odd sf_k (padding path), 2-D and 3-D (expert-batched) inputs.
"""

import hashlib
import json
import sys

import torch


def digest(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        # logical values in row order — the layout strides are compared separately
        "md5": hashlib.md5(t.contiguous().cpu().view(torch.int32).numpy().tobytes()).hexdigest(),
    }


def main() -> None:
    from vllm.utils.deep_gemm import transform_sf_into_required_layout

    torch.manual_seed(20260811)
    cases = []

    def ue8m0_class(*shape):
        # Production input class: `_upcast_e8m0_to_fp32` output = pure powers of two
        # (exponent from the checkpoint's UE8M0 bytes, mantissa exactly zero). Random
        # mantissas are NOT a valid input here — the DG JIT kernel's overlapping-OR
        # packing only coincides with the clean packing when mantissas are zero, which
        # the ue8m0 checkpoint flow guarantees (scale_fmt=ue8m0 detected at boot).
        exps = torch.randint(96, 154, shape, dtype=torch.int32)
        return torch.ldexp(torch.ones(shape), exps - 127).to(torch.float32).cuda()

    # (mn, sf_k) sweep: multiples of 4, odd mn, odd sf_k, degenerate
    for mn, sk in [
        (2048, 16),
        (7168, 32),
        (4096, 56),
        (8192, 18),
        (1023, 7),
        (255, 33),
        (128, 4),
        (4, 1),
        (1, 1),
    ]:
        sf = ue8m0_class(mn, sk)
        # 2-D sf pairs with num_groups=None (C++: sf.dim() == num_groups.has_value() + 2)
        cases.append({"sf": sf, "mn": mn, "k": sk * 128, "num_groups": None, "tag": f"2d-{mn}x{sk}"})
    # 3-D expert-batched cases (num_groups > 1)
    for g, mn, sk in [(4, 512, 24), (3, 100, 5), (8, 2048, 16)]:
        sf = ue8m0_class(g, mn, sk)
        cases.append({"sf": sf, "mn": mn, "k": sk * 128, "num_groups": g, "tag": f"3d-{g}x{mn}x{sk}"})

    for c in cases:
        out = transform_sf_into_required_layout(
            sf=c["sf"],
            mn=c["mn"],
            k=c["k"],
            recipe=(1, 1, 128),
            num_groups=c["num_groups"],
            is_sfa=False,
        )
        print(json.dumps({"tag": c["tag"], **digest(out)}))


if __name__ == "__main__":
    sys.exit(main())
