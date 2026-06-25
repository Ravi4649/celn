"""
Ablation Study: ProofWriter Accuracy vs Dimensionality (D)
===========================================================
Testa D = 10000, 5000, 1000 em 100 exemplos do ProofWriter real.
Gera ablation_results.tex com tabela LaTeX.

Uso:
    python experiments/ablation_dimensions.py

Requer:
    pip install datasets
    HF_TOKEN com acesso a tasksource/proofwriter
"""

import subprocess, sys, json, time, os, tempfile, textwrap
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
EXPERIMENTS = str(Path(__file__).resolve().parent)
DIMS = [10000, 5000, 1000]
N_EXAMPLES = 100

RUNNER_TEMPLATE = '''\
import sys, json, time, re, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, {root!r})

# 1. PATCH dimensionalidade ANTES de qualquer import celn_v3
import celn_v3.core
celn_v3.core.D = {dim}

# 2. Agora importa o benchmark (pega D já patched)
from experiments.benchmark_proofwriter_real import run_benchmark

t0 = time.time()
results = run_benchmark({n})
elapsed = time.time() - t0

total = len(results)
correct = sum(1 for r in results if r["correct"])
by_label = {{}}
for label in ["True", "False", "Unknown"]:
    subset = [r for r in results if r["gold"] == label]
    if subset:
        ok = sum(1 for r in subset if r["correct"])
        by_label[label] = {{"correct": ok, "total": len(subset), "acc": round(ok / len(subset), 4)}}

parsed = sum(1 for r in results if r.get("parsed"))
parsed_ok = sum(1 for r in results if r.get("parsed") and r["correct"])

stats = {{
    "dim": {dim},
    "total": total,
    "correct": correct,
    "accuracy": round(correct / total, 4),
    "parsed": parsed,
    "parsed_ok": parsed_ok,
    "parsed_accuracy": round(parsed_ok / parsed, 4) if parsed else 0,
    "by_label": by_label,
    "time_s": round(elapsed, 2),
}}
print(json.dumps(stats))
'''

def run_dim(dim: int) -> dict:
    """Executa benchmark em subprocesso com dimensionalidade D = dim."""
    code = RUNNER_TEMPLATE.format(root=ROOT, dim=dim, n=N_EXAMPLES)
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, prefix=f'ablation_d{dim}_'
    ) as f:
        f.write(code)
        runner_path = f.name

    try:
        print(f"  D={dim:5d}... ", end='', flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True, text=True, timeout=600,
        )
        wall = time.time() - t0

        if proc.returncode != 0:
            print(f"FAILED (rc={proc.returncode})")
            print("STDERR:", proc.stderr[:500])
            return None

        stats = json.loads(proc.stdout.strip())
        stats['wall_s'] = round(wall, 2)
        print(f"acc={stats['accuracy']:.1%}  parsed={stats['parsed_accuracy']:.1%}  "
              f"({stats['time_s']}s cpu / {wall}s wall)")
        return stats

    except subprocess.TimeoutExpired:
        print("TIMEOUT (>600s)")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}")
        print("STDOUT:", proc.stdout[:500] if hasattr(proc, 'stdout') else '(none)')
        return None
    finally:
        os.unlink(runner_path)


def latex_table(stats_dict: dict) -> str:
    """Gera tabela LaTeX a partir dos resultados."""
    rows = []
    for dim in DIMS:
        s = stats_dict.get(dim)
        if s is None:
            rows.append(
                f"  {dim} & --- & --- & --- & --- & --- \\\\"
            )
            continue
        b = s['by_label']
        rows.append(
            f"  {dim} & {s['correct']}/{s['total']} & {s['accuracy']:.1%} & "
            f"{b.get('True', {}).get('correct', 0)}/{b.get('True', {}).get('total', 0)} & "
            f"{b.get('False', {}).get('correct', 0)}/{b.get('False', {}).get('total', 0)} & "
            f"{b.get('Unknown', {}).get('correct', 0)}/{b.get('Unknown', {}).get('total', 0)} \\\\"
        )

    tex = textwrap.dedent(f"""\
    % Gerado por experiments/ablation_dimensions.py em {time.strftime('%Y-%m-%d %H:%M')}
    % N = {N_EXAMPLES} exemplos do ProofWriter (tasksource/proofwriter)
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Ablação da dimensionalidade $D$ no ProofWriter ($n={N_EXAMPLES}$)}}
    \\label{{tab:ablation_dim}}
    \\begin{{tabular}}{{lrrrrr}}
    \\toprule
    $D$ & \\multicolumn{{1}}{{c}}{{Total}} & \\multicolumn{{1}}{{c}}{{Acurácia}} & \\multicolumn{{1}}{{c}}{{True}} & \\multicolumn{{1}}{{c}}{{False}} & \\multicolumn{{1}}{{c}}{{Unknown}} \\\\
    \\midrule
    {chr(10).join(rows)}
    \\bottomrule
    \\end{{tabular}}
    \\end{{table}}
    """)
    return tex


def main():
    print("=" * 60)
    print("Ablation: ProofWriter Accuracy vs Dimensionality")
    print(f"  Dimensões: {DIMS}")
    print(f"  Exemplos:  {N_EXAMPLES}")
    print("=" * 60)

    all_stats = {}
    for dim in DIMS:
        stats = run_dim(dim)
        all_stats[dim] = stats
        print()

    # Gera tabela LaTeX
    tex = latex_table(all_stats)
    tex_path = Path(EXPERIMENTS) / '..' / 'ablation_results.tex'
    tex_path.write_text(tex, encoding='utf-8')
    print(f"Tabela LaTeX salva em: {tex_path.resolve()}")
    print(tex)


if __name__ == '__main__':
    main()
