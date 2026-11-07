"""Patch vLLM 0.26 so Gemma 4's per-layer embedding (PLE) table can live in host memory.

1. gemma4.py: the PLE table is built outside make_layers, so the offloader never saw it and
   `--cpu-offload-params embed_tokens_per_layer` matched nothing. Wrap it when the offloader is UVA
   (upstream PR vllm-project/vllm#62354, ported to 0.26's offloader API).
2. V2 model runner: 0.26 never installs the offloader there (only the V1 runner does), so every
   offload flag is a no-op for the models it runs, Gemma 4 included. Main does this since 0.32.

Every edit asserts its anchor appears exactly once, so a different vLLM fails the build instead of
silently skipping a patch.
"""
import pathlib
import sys

import vllm

ROOT = pathlib.Path(vllm.__file__).parent


def edit(rel, old, new):
    p = ROOT / rel
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"patch_ple: {rel}: anchor found {n} times, expected 1:\n{old}")
    p.write_text(s.replace(old, new))
    print(f"patch_ple: {rel} patched")


edit("model_executor/models/gemma4.py",
     "from vllm.platforms import current_platform\n",
     "import contextlib\n\n"
     "from vllm.model_executor.offloader import get_offloader\n"
     "from vllm.model_executor.offloader.uva import UVAOffloader\n"
     "from vllm.platforms import current_platform\n"
     "from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor\n")

edit("model_executor/models/gemma4.py",
     """            total_ple_dim = self.hidden_size_per_layer_input * config.num_hidden_layers
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                self.vocab_size_per_layer_input,
                total_ple_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens_per_layer",
            )
""",
     """            total_ple_dim = self.hidden_size_per_layer_input * config.num_hidden_layers
            # The PLE table is built directly, so make_layers never routes it through the
            # offloader. With UVA and `--cpu-offload-params embed_tokens_per_layer`, build it
            # on the CPU and give the GPU a view of its pinned memory: each token gathers only
            # its own rows. Building it on the GPU first and copying it out (what wrap_modules
            # does) holds 5.25 GiB of GPU memory during the load for nothing.
            offloader = get_offloader()
            ple_uva = (isinstance(offloader, UVAOffloader) and offloader.uva_offloading
                       and "embed_tokens_per_layer" in offloader.cpu_offload_params)
            with torch.device("cpu") if ple_uva else contextlib.nullcontext():
                self.embed_tokens_per_layer = VocabParallelEmbedding(
                    self.vocab_size_per_layer_input,
                    total_ple_dim,
                    quant_config=quant_config,
                    prefix=f"{prefix}.embed_tokens_per_layer",
                )
            if ple_uva:
                for p in self.embed_tokens_per_layer.parameters():
                    host = p.data.pin_memory() if offloader.pin_memory else p.data
                    p.data = get_accelerator_view_from_cpu_tensor(host)
                    p._vllm_is_uva_offloaded = True
                    offloader.cpu_offload_bytes += host.numel() * host.element_size()
                logger.info("Gemma 4 per-layer embeddings in pinned host memory: %.2f GiB",
                            offloader.cpu_offload_bytes / 2**30)
""")

edit("v1/worker/gpu/model_runner.py",
     "from vllm.model_executor.model_loader import get_model_loader\n",
     "from vllm.model_executor.model_loader import get_model_loader\n"
     "from vllm.model_executor.offloader import create_offloader, get_offloader, set_offloader\n")

edit("v1/worker/gpu/model_runner.py",
     """        self.eplb = EPLBController(self.parallel_config, self.device)

    def update_max_model_len(""",
     """        self.eplb = EPLBController(self.parallel_config, self.device)

        set_offloader(create_offloader(self.vllm_config.offload_config))

    def update_max_model_len(""")

edit("v1/worker/gpu/model_runner.py",
     """        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory""",
     """        get_offloader().post_init()
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory""")
