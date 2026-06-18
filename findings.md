## Findings
- Current `projective_resonance` emits an opaque sequential scan state, not a port-addressed state.
- Existing `bind`, `unbind`, `make_random_vector`, and `D=10000` in `celn_v3/core.py` are sufficient to create anonymous VSA ports.
- `corpus_final.txt` and existing lexical vectors can calibrate empirical sensor distributions.
- Unit-magnitude Fourier ports make known-address `unbind(bind(PORT, value), PORT)` stable enough for register readback.
- With 32 ports and 160 calibration sentences, M_ctrl readback median sentence MAE was `0.012418`, far below the null register alignment p10 `0.286040`.
- GPVE generation works mechanically with PortAdapter registers and PCFG rule reweighting, but semantic conditioning is still weak for some prompts because `pcfg_pruned.json` has limited start-symbol coverage and high-probability continuations mix domains.
- The GPVE implementation does not call 10k cosine similarity or vocabulary matrix scoring during generation.
- Phase Lens changes GPVE control bins but did not reliably improve semantic binding; next signal source should factor relation-like components before PortAdapter instead of only changing phase globally.
