"""
probes/bench_latency_memory.py - P8/P16 batch=1, warm-up 이후 기준
latency(mean+-std)/peak memory/params/FLOPs 측정. (탐색적, 롤백 가능)

방법
----
- 각 모델을 warm-up(기본 20회) 돌린 뒤, torch.cuda.synchronize()로 정확히 시간 잰
  100회 반복(batch=1)의 mean+-std latency
- torch.cuda.reset_peak_memory_stats() 이후 max_memory_allocated()로 peak memory
- 파라미터 수는 단순 합산
- FLOPs는 torch 내장 FlopCounterMode 사용 (fvcore/thop 미설치라 이걸로 대체, 로컬 API 검증 완료)

주의: src/models.py는 import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python probes/bench_latency_memory.py --n_warmup 20 --n_iters 100
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models import get_device, load_vit_model


def benchmark(model, device, n_warmup, n_iters):
    x = torch.randn(1, 3, 224, 224, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        for _ in range(n_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                       if device.type == 'cuda' else float('nan'))

    times = np.array(times)
    params = sum(p.numel() for p in model.parameters())

    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            model(x)
    flops = fcm.get_total_flops()

    return dict(mean_ms=times.mean(), std_ms=times.std(), median_ms=np.median(times),
               peak_mem_mb=peak_mem_mb, params=params, flops=flops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_warmup', type=int, default=20)
    parser.add_argument('--n_iters', type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    results = {}
    for P in (8, 16):
        print(f"\n=== P={P} ===")
        model = load_vit_model(P, device)
        r = benchmark(model, device, args.n_warmup, args.n_iters)
        results[P] = r
        print(f"  latency (batch=1) : {r['mean_ms']:.3f} +- {r['std_ms']:.3f} ms "
              f"(median {r['median_ms']:.3f} ms, n={args.n_iters})")
        print(f"  peak memory       : {r['peak_mem_mb']:.1f} MB")
        print(f"  params            : {r['params']/1e6:.2f} M")
        print(f"  FLOPs (batch=1)   : {r['flops']/1e9:.3f} GFLOPs")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'P':>4} {'latency(ms)':>16} {'peak_mem(MB)':>14} {'params(M)':>11} {'GFLOPs':>10}")
    print("-" * 60)
    for P, r in results.items():
        print(f"{P:>4} {r['mean_ms']:>8.3f}+-{r['std_ms']:<6.3f} {r['peak_mem_mb']:>14.1f} "
              f"{r['params']/1e6:>11.2f} {r['flops']/1e9:>10.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, '10_bench_latency_memory.npz'),
             **{f'P{P}_{k}': v for P, r in results.items() for k, v in r.items()})
    print(f"\nSaved: {os.path.join(out_dir, '10_bench_latency_memory.npz')}")


if __name__ == '__main__':
    main()
