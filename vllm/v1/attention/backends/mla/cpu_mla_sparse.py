# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V3.2-style sparse MLA (DSA) in a simulation on CPU: the GPU
deployment's FlashMLA sparse backend, its KV cache spec, layout and block
sizes, with its kernels never run (a simulation executes no forward)."""

from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend


class CPUSimulatedMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "CPU_SIMULATED_MLA_SPARSE"
