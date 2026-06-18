## Goal
Implement a non-metric PortAdapter for CELN v3 that converts opaque projective-resonance states into addressable control states for GPVE.

## Phases
- Status: complete | Phase 1: Inspect current M/corpus/PCFG contracts.
- Status: complete | Phase 2: Implement PortAdapter and non-metric reader.
- Status: complete | Phase 3: Add a functional verification script/test.
- Status: complete | Phase 4: Run verification and record results.

## Constraints
- No backpropagation, transformers, fixed word lists, templates, or magic thresholds.
- No cosine or distance in 10k dimensions for text generation.
- Calibration uses empirical percentiles from corpus-derived M states.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `ModuleNotFoundError: No module named 'celn_v3'` running `experiments/test_port_adapter.py` directly | 1 | Added project-root bootstrap to the experiment script. |

## Verification Results
- `python -m compileall celn_v3/port_adapter.py experiments/test_port_adapter.py` passed.
- `python experiments/test_port_adapter.py --calib-sentences 160 --eval-sentences 48 --ports 32` passed with mean register read error `0.012009` and max `0.064287`.
- Save/load smoke check passed from `/tmp/opencode/port_adapter_state.npz` with readback MAE `0.004900`.

## GPVE Extension
- Status: complete | Phase 5: Implement GPVE mouth using PortAdapter registers and PCFG rule reweighting.
- Status: complete | Phase 6: Run generation experiment and record real output.

## GPVE Verification Results
- `python -m compileall celn_v3/gpve_mouth.py experiments/run_gpve.py` passed.
- Grep found no calls to `similarity`, `cosine_similarity`, or vocabulary matrix scoring in `celn_v3/gpve_mouth.py` or `experiments/run_gpve.py`.
- Final run: `python experiments/run_gpve.py --adapter-sentences 160 --ports 32 --max-tokens 24 --greedy Os gatos possuem um cérebro bastante evoluído sendo capazes de sentir emoções`.
- Final output: `As formigas são animais muitos outros seres heterotróficos seriam incapazes de sobreviver`.

## IntentDistiller Extension
- Status: in_progress | Phase 7: Implement IntentDistiller using Phase Lens, Resonator extraction, and transport-style canonicalization.
- Status: pending | Phase 8: Integrate IntentDistiller into GPVE/runner behind explicit flags.
- Status: pending | Phase 9: Verify compile/tests and compare the four prompts against baseline/Phase Lens.
