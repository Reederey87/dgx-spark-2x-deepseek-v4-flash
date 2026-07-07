# Credits & provenance

This repository is **orchestration and documentation** — a reproducible, sanitized
harness for a deployment that stands entirely on the work of the community authors
below. It vendors none of their source: the serving image is **built from** the
pinned recipe and the weights are **pulled from** Hugging Face at deploy time. If
you use this kit, credit them; if you improve on it, send fixes upstream.

> ⚠️ **Everything in the DSpark / GB10 serving stack is fast-moving, largely
> single-author, and partly dependent on prebuilt (non-source-buildable) kernels
> and images. Treat it as experimental and validate on your own hardware behind
> each project's own smoke/sanity tests.**

## The serving recipe (what this kit pins and builds)

| Author | What | Link |
|---|---|---|
| **tonyd2wild** | The 2× DGX Spark NVFP4-KV + DSpark serving recipe this kit's `docker-compose.dspark.yml` is vendored from (pinned at `RECIPE_SHA` in `cluster.env`). | https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark |
| **drowzeys ("Keys")** | The DSpark **scalable-concurrency patch**, the `nvfp4_ds_mla` KV path, the **"dual cache"** sparse-MLA split-K fix, and the vLLM 0.24.0 optimization write-up. | https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash · https://github.com/drowzeys/keys-vLLm-0.24.0-Optimized-DeepSeekV4-Flash-DSpark-NVFP4-KV-1.5M-CTX-3M-Pool-C-12-on-2-DGX-Spark |
| **aidendle94** | The compiled GB10 (`sm_121a`) **DeepGEMM** and **sparse-MLA** kernels transplanted into the serving image (not public source). | (kernels ship inside the base/stage image) |
| **MiaAI-Lab** | Dual-Spark packaging, worker-before-head launch ordering. | https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark |
| **rafaelcaricio** | DSpark vLLM integration. | https://github.com/rafaelcaricio/vllm |

## Foundations

| Author | What | Link |
|---|---|---|
| **DeepSeek AI** | `DeepSeek-V4-Flash-DSpark` model weights (license per the model card). | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark |
| **vLLM** | The inference engine (Apache-2.0); DSpark landed upstream in PR #46995. | https://github.com/vllm-project/vllm |
| **NVIDIA** | DGX Spark, the GB10 platform, and the official `dgx-spark-playbooks` "connect two Sparks" guidance this networking setup follows. | https://github.com/NVIDIA/dgx-spark-playbooks |

## Base image

`ghcr.io/bjk110/vllm-spark` (pinned by immutable digest in `cluster.env`) is a
prebuilt community base for GB10. It is **not source-auditable**; this kit pins it
by digest, runs it non-privileged (`--gpus all` + `/dev/infiniband` device only),
binds the API to loopback, and mounts no secrets. See `docs/03-model-and-features.md`
for the full provenance chain and `docs/05-troubleshooting.md` for the security posture.
