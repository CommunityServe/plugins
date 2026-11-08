# turboexl3_ds41

Serves DeepSeek-V4.1-Flash's exllamav3 (EXL3) packs, such as the streaming repack
(https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw), as a quant plugin.
mlx-serve's own `deepseek_v41` serves the pack: its forward, DSpark, the prefix cache and the bill. This plugin claims
the pack's `quantization_config` (`quant_method` exl3), runs the weights the pack stores in it on turboexl3's Metal
kernels, and streams the routed experts from the repack's `experts.bin` (a4k layout included):

- every trellis linear of the trunk, the LM head and the DSpark stages (`matmul`: turboexl3's `linear`; a prompt's
  rows on its fused matmul + finish, pg2), the grouped `wo_a` as one weight (`src/woa_join.zig` at prompt rows);
- the shared expert at a verify's rows in one launch (`swiglu` with no routes, `src/shared_swiglu.zig`);
- the DSpark stages' experts, each stage's per-expert linears stacked into one bank at load (`swiglu`: turboexl3's
  `moeClamped`, the pack's `swiglu_limit`);
- the streamed bank (`bank_rows` / `open_bank`, an `abi.Bank` the trunk reads as its routed experts): the EXL3 lane
  over the kit's expert stream (`kit.streamed.expert`: slot pools, residency policy, the read pool, whose C the kit's
  `streamSources` compiles in), as this plugin's own source (module `mlx_stream`, `src/stream.zig`: `src/streamed/`
  the EXL3 bank, its source and waves over its own `sdk`; `src/exl3/` the prompt lane's kernels; `src/bank/`
  DeepSeek-V4.1's schedule over them), its decode waves on turboexl3's grouped banked decode, any layer at
  exllamav3's half-integer rates on turboexl3's MoE. MLX's internals the stream needs (events, buffer aliases, the
  launch cache, the command-buffer split) come through the host table (`src/host_shims.zig`).

The bank's prompt lane needs an M5-class GPU (NAX; elsewhere the bank is refused by name). Images are not served on
these packs yet (the vision tower is EXL3 too). The bank reads `TURBOEXL3_DS41_FRAC`, `TURBOEXL3_DS41_RESIDENT_SEED`,
`TURBOEXL3_DS41_EXPERT_USAGE_{DIR,RECORD,LABEL}` (its options: the app's Models ▸ Plugins page sets them) and
`MLX_MAX_MB_PER_BUFFER` at its open.

    scripts/build.sh turboexl3_ds41                    # from the repo root: tests, then ReleaseFast
    scripts/install.sh turboexl3_ds41 --home ~/tmp/h   # under a throwaway HOME

Needs an mlx-serve whose quant table has `open_bank` (`sdk/quant/outside.zig`). The log names the plugin at load:
`deepseek_v41: lm head on turboexl3_ds41, as shipped`, `... weights on turboexl3_ds41` and `deepseek_v41: streamed
experts, N prompt / M decode slot rows per layer`.
