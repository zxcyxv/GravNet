"""
analyze_gnorm.py — gnorm_per_loop.jsonl 분석

사용법:
  python analyze_gnorm.py                            # 기본 (gnorm_per_loop.jsonl)
  python analyze_gnorm.py --file other.jsonl
  python analyze_gnorm.py --spike_thresh 5.0        # 스파이크 임계값 조정
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path: str):
    return [json.loads(l) for l in open(path)]


def stats(vals):
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
    s  = sorted(vals)
    mu = sum(vals) / n
    std = math.sqrt(sum((x - mu)**2 for x in vals) / max(1, n-1))
    return {
        "n":   n,
        "mean": mu,
        "std":  std,
        "min":  s[0],
        "max":  s[-1],
        "p50":  s[int(n * 0.50)],
        "p95":  s[int(n * 0.95)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file",         default="gnorm_per_loop.jsonl")
    p.add_argument("--spike_thresh", type=float, default=5.0,
                   help="TOTAL_pre_clip 이상이면 스파이크로 간주")
    args = p.parse_args()

    records = load(args.file)
    print(f"총 {len(records)} 스텝 로드\n")

    num_layers = max(
        int(k.split("_")[0][1:]) + 1
        for r in records for k in r if k.startswith("b") and "_TOTAL" in k
    )

    # ── 1. 루프별 gnorm 통계 ──────────────────────────────────────────────
    by_loop: dict = defaultdict(list)
    for r in records:
        loop = int(round(r["carry_steps_mean"]))
        by_loop[loop].append(r["TOTAL_pre_clip"])

    print("=" * 70)
    print("1. 루프별 TOTAL gnorm 통계")
    print("=" * 70)
    print(f"{'loop':>5} {'n':>5} {'mean':>8} {'std':>8} {'p50':>8} {'p95':>8} {'max':>8}")
    print("-" * 70)
    for loop in sorted(by_loop):
        s = stats(by_loop[loop])
        print(f"{loop:>5} {s['n']:>5} {s['mean']:>8.3f} {s['std']:>8.3f} "
              f"{s['p50']:>8.3f} {s['p95']:>8.3f} {s['max']:>8.3f}")

    # ── 2. 컴포넌트별 기여도 분석 ─────────────────────────────────────────
    comp_keys = (
        [f"b{i}_QKV" for i in range(num_layers)] +
        [f"b{i}_O"   for i in range(num_layers)] +
        [f"b{i}_up"  for i in range(num_layers)] +
        [f"b{i}_dn"  for i in range(num_layers)] +
        ["embed", "lm_head"]
    )

    print("\n" + "=" * 70)
    print("2. 컴포넌트별 gnorm 통계 (전체 스텝)")
    print("=" * 70)
    print(f"{'component':>15} {'mean':>8} {'std':>8} {'p95':>8} {'max':>8}")
    print("-" * 70)

    comp_stats = {}
    for k in comp_keys:
        vals = [r[k] for r in records if k in r]
        if not vals:
            continue
        s = stats(vals)
        comp_stats[k] = s
        print(f"{k:>15} {s['mean']:>8.3f} {s['std']:>8.3f} {s['p95']:>8.3f} {s['max']:>8.3f}")

    # ── 3. 스파이크 발생 시 컴포넌트 비중 ────────────────────────────────
    spikes  = [r for r in records if r["TOTAL_pre_clip"] >= args.spike_thresh]
    normals = [r for r in records if r["TOTAL_pre_clip"] <  args.spike_thresh]

    print(f"\n{'='*70}")
    print(f"3. 스파이크(TOTAL≥{args.spike_thresh}) vs 정상 구간 비교")
    print(f"   스파이크 {len(spikes)}건 / 정상 {len(normals)}건")
    print(f"{'='*70}")
    print(f"{'component':>15} {'spike_mean':>12} {'normal_mean':>12} {'배율':>8}")
    print("-" * 70)
    for k in comp_keys:
        if k not in comp_stats:
            continue
        sp = sum(r[k] for r in spikes)  / max(1, len(spikes))
        nm = sum(r[k] for r in normals) / max(1, len(normals))
        ratio = sp / nm if nm > 0 else float("inf")
        print(f"{k:>15} {sp:>12.3f} {nm:>12.3f} {ratio:>8.1f}×")

    # ── 4. 스파이크 상세 목록 ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"4. 스파이크 발생 목록 (TOTAL≥{args.spike_thresh})")
    print(f"{'='*70}")
    print(f"{'step':>6} {'loop':>5} {'TOTAL':>8}  최대 컴포넌트")
    print("-" * 70)

    for r in sorted(spikes, key=lambda x: -x["TOTAL_pre_clip"]):
        loop = int(round(r["carry_steps_mean"]))
        total = r["TOTAL_pre_clip"]
        # 최대 컴포넌트 상위 3개
        ranked = sorted(
            [(k, r[k]) for k in comp_keys if k in r],
            key=lambda x: -x[1]
        )[:3]
        top = "  ".join(f"{k}={v:.2f}" for k, v in ranked)
        print(f"{r['step']:>6} {loop:>5} {total:>8.3f}  {top}")

    # ── 5. 루프 번호와 스파이크 관계 ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("5. 루프별 스파이크 발생률")
    print(f"{'='*70}")
    spike_by_loop: dict = defaultdict(int)
    count_by_loop: dict = defaultdict(int)
    for r in records:
        loop = int(round(r["carry_steps_mean"]))
        count_by_loop[loop] += 1
        if r["TOTAL_pre_clip"] >= args.spike_thresh:
            spike_by_loop[loop] += 1

    print(f"{'loop':>5} {'total':>7} {'spikes':>8} {'rate':>8}  bar")
    print("-" * 70)
    for loop in sorted(count_by_loop):
        cnt = count_by_loop[loop]
        sp  = spike_by_loop[loop]
        rate = sp / cnt if cnt > 0 else 0
        bar  = "#" * int(rate * 20)
        print(f"{loop:>5} {cnt:>7} {sp:>8} {rate:>7.1%}  {bar}")

    # ── 6. 블록별 QKV 평균 gnorm ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("6. 블록별 QKV gnorm 평균 (전체 / 스파이크 시)")
    print(f"{'='*70}")
    print(f"{'block':>7} {'전체평균':>10} {'스파이크':>10} {'정상':>10}")
    print("-" * 70)
    for i in range(num_layers):
        k = f"b{i}_QKV"
        all_  = sum(r[k] for r in records if k in r) / max(1, len(records))
        sp_m  = sum(r[k] for r in spikes)  / max(1, len(spikes))
        nm_m  = sum(r[k] for r in normals) / max(1, len(normals))
        print(f"{'b'+str(i)+' QKV':>7} {all_:>10.3f} {sp_m:>10.3f} {nm_m:>10.3f}")


if __name__ == "__main__":
    main()
