#!/usr/bin/env python3
"""Run the GPVE mouth and print real generated text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celn.gpve_mouth import GPVEMouth

PROMPTS_PT = [
    "Na floresta fria o lobo uivou sozinho",
    "O cientista descobriu uma nova especie de borboleta",
    "A cidade antiga foi construida sobre as ruinas de um imperio",
    "O vento soprava forte enquanto o navio enfrentava as ondas",
]


def _print_debug_log(log, title=""):
    if not log:
        return
    print(f"\n--- {title} ({len(log)} steps) ---")
    header = f"{'Step':>4} | {'LHS':<15} | {'Source':<25} | {'Dominant':<18} | {'Token':<15} | PCFG Anch Mag Ph Typ SDM Tra spa"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for d in log:
        lhs = d.get('lhs', '?')[-15:]
        src = d.get('source', '?')
        dom = f"{d.get('dominant_channel','?')} {d.get('dominant_val',0):.2f}" if d.get('dominant_channel') else '?'
        tok = (d.get('winner_token') or '?')[:15]
        ch = d.get('weighted_ch', {})
        p = ch.get('PCFG', 0)
        a = ch.get('Anchor', 0)
        m = ch.get('Magnitude', 0)
        ph = ch.get('Phase', 0)
        ty = ch.get('Type', 0)
        s = ch.get('SDM', 0)
        tr = ch.get('Trajectory', 0)
        sp = ch.get('spaCy', 0)
        print(f"{d['step']:>4} | {lhs:<15} | {src:<25} | {dom:<18} | {tok:<15} | {p:.2f} {a:.2f} {m:.2f} {ph:.2f} {ty:.2f} {s:.2f} {tr:.2f} {sp:.2f}")


def _count_dominant(log, channel):
    return sum(1 for d in log if d.get('dominant_channel') == channel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with GPVE")
    parser.add_argument("--pcfg", default="pcfg_pruned.json")
    parser.add_argument("--vectors", default="celn_full_vectors.npz")
    parser.add_argument("--corpus", default="corpus_final.txt")
    parser.add_argument("--adapter-sentences", type=int, default=256)
    parser.add_argument("--ports", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--token-min-len", type=int, default=2)
    parser.add_argument("--phase-lens-alpha", type=float, default=0.5)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--debug-pcfg", action="store_true",
                        help="Print per-step PCFG decision debug for 4 standard prompts")
    parser.add_argument("--structure-only", action="store_true",
                        help="PCFG as syntactic skeleton — no PCFG vote")
    args = parser.parse_args()

    adapter_sentences = None if args.adapter_sentences <= 0 else args.adapter_sentences
    prompts_to_run = PROMPTS_PT if args.debug_pcfg or args.structure_only else [" ".join(args.prompt).strip() or "Na floresta fria o lobo uivou sozinho"]

    # Build mouth(s) once, reuse for all prompts
    if args.structure_only:
        mouth_default = GPVEMouth.build(
            pcfg_path=args.pcfg, vectors_path=args.vectors, corpus_path=args.corpus,
            adapter_max_sentences=adapter_sentences, n_ports=args.ports,
            seed=args.seed, max_tokens=args.max_tokens, token_min_len=args.token_min_len,
            rule_calibration_corpus=args.corpus, use_phase_lens=True,
            phase_lens_alpha=args.phase_lens_alpha, use_intent_distiller=False,
            structure_only=False,
        )
        mouth_struct = GPVEMouth.build(
            pcfg_path=args.pcfg, vectors_path=args.vectors, corpus_path=args.corpus,
            adapter_max_sentences=adapter_sentences, n_ports=args.ports,
            seed=args.seed, max_tokens=args.max_tokens, token_min_len=args.token_min_len,
            rule_calibration_corpus=args.corpus, use_phase_lens=True,
            phase_lens_alpha=args.phase_lens_alpha, use_intent_distiller=False,
            structure_only=True,
        )
        if args.debug_pcfg:
            mouth_default.debug_pcfg_enabled = True
            mouth_struct.debug_pcfg_enabled = True

        for pi, prompt_text in enumerate(prompts_to_run):
            if args.debug_pcfg:
                mouth_default._debug_pcfg_log = []
                mouth_struct._debug_pcfg_log = []
            out_default = mouth_default.generate_from_text(prompt_text, sample=not args.greedy, max_tokens=args.max_tokens)
            out_struct = mouth_struct.generate_from_text(prompt_text, sample=not args.greedy, max_tokens=args.max_tokens)

            print(f"\n{'='*70}")
            print(f"Prompt {pi+1}: {prompt_text}")
            print(f"{'='*70}")
            print(f"  BEFORE (PCFG vota):   {out_default}")
            print(f"  AFTER  (só esqueleto): {out_struct}")

            if args.debug_pcfg:
                ld = mouth_default._debug_pcfg_log or []
                ls = mouth_struct._debug_pcfg_log or []
                p_before = 100 * _count_dominant(ld, 'PCFG') / max(1, len(ld))
                p_after = 100 * _count_dominant(ls, 'PCFG') / max(1, len(ls))
                print(f"\n  ▶ PCFG domina: {p_before:.0f}% → {p_after:.0f}%")
                _print_debug_log(ld, "ANTES")
                _print_debug_log(ls, "DEPOIS")
    else:
        mouth = GPVEMouth.build(
            pcfg_path=args.pcfg, vectors_path=args.vectors, corpus_path=args.corpus,
            adapter_max_sentences=adapter_sentences, n_ports=args.ports,
            seed=args.seed, max_tokens=args.max_tokens, token_min_len=args.token_min_len,
            rule_calibration_corpus=args.corpus, use_phase_lens=True,
            phase_lens_alpha=args.phase_lens_alpha, use_intent_distiller=False,
        )
        if args.debug_pcfg:
            mouth.debug_pcfg_enabled = True

        for pi, prompt_text in enumerate(prompts_to_run):
            if args.debug_pcfg:
                mouth._debug_pcfg_log = []
            print(f"\n{'='*70}")
            print(f"Prompt {pi+1}: {prompt_text}")
            print(f"{'='*70}")
            out = mouth.generate_from_text(prompt_text, sample=not args.greedy, max_tokens=args.max_tokens)
            print(f"GPVE+PhaseLens: {out}")
            if args.debug_pcfg and mouth._debug_pcfg_log:
                _print_debug_log(mouth._debug_pcfg_log, "DEBUG")


if __name__ == "__main__":
    main()
