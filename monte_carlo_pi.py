import math
import time
import argparse
import numpy as np


# ── Core simulations ────────────────────────────────────────────────────────

def estimate_pi_numpy(num_samples: int, seed: int = None) -> tuple[float, list]:
    """
    Vectorized Monte Carlo π estimation using NumPy.

    Strategy:
      - Inscribe a unit circle (radius=1) in a 2x2 square.
      - Generate all (x, y) points at once as NumPy arrays.
      - A point is inside the circle if x^2 + y^2 <= 1.
      - pi ≈ 4 * (points inside circle) / (total points)

    Much faster than the pure-Python version because:
      - np.random generates all samples in one C-level call (no Python loop).
      - The distance check is a vectorized array operation (no per-point branching).
      - NumPy's BLAS-backed math runs in parallel under the hood.

    Returns:
        (pi_estimate, running_estimates)
        running_estimates is a list of (sample_number, pi_estimate) tuples
        at logarithmic checkpoints for plotting convergence.
    """
    rng = np.random.default_rng(seed)

    # Generate all points at once — shape (2, num_samples)
    points = rng.uniform(-1.0, 1.0, size=(2, num_samples))

    # Vectorized distance check: True where x^2 + y^2 <= 1
    inside_mask = (points[0] ** 2 + points[1] ** 2) <= 1.0

    # Cumulative sum gives "inside count so far" at every index — no Python loop
    cumulative_inside = np.cumsum(inside_mask)

    # Build logarithmic checkpoints
    checkpoints = _log_checkpoints(num_samples)
    idx = np.array(checkpoints) - 1  # 0-based indices
    running_estimates = [
        (int(i + 1), float(4 * cumulative_inside[i] / (i + 1)))
        for i in idx
    ]

    pi_estimate = float(4 * cumulative_inside[-1] / num_samples)
    return pi_estimate, running_estimates


def estimate_pi_pure(num_samples: int, seed: int = None) -> tuple[float, list]:
    """
    Pure-Python Monte Carlo π estimation (kept for benchmarking comparison).
    Uses the standard library random module with an explicit Python loop.
    """
    import random
    if seed is not None:
        random.seed(seed)

    inside = 0
    running_estimates = []
    checkpoints = set(_log_checkpoints(num_samples))

    for i in range(1, num_samples + 1):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            inside += 1
        if i in checkpoints:
            running_estimates.append((i, 4 * inside / i))

    return 4 * inside / num_samples, running_estimates


def _log_checkpoints(num_samples: int) -> list[int]:
    """Return a sorted list of logarithmically-spaced sample indices (1-based)."""
    checkpoints = set()
    n = 1
    while n <= num_samples:
        checkpoints.add(n)
        n = max(n + 1, int(n * 1.1))
    checkpoints.add(num_samples)
    return sorted(checkpoints)


# ── Output helpers ───────────────────────────────────────────────────────────

def print_results(num_samples: int, pi_estimate: float,
                  running_estimates: list, elapsed: float = None,
                  label: str = "NumPy"):
    error = abs(pi_estimate - math.pi)
    error_pct = error / math.pi * 100

    print(f"\n{'='*52}")
    print(f"  Monte Carlo π  [{label}]")
    print(f"{'='*52}")
    print(f"  Samples:      {num_samples:,}")
    print(f"  Estimated π:  {pi_estimate:.6f}")
    print(f"  True π:       {math.pi:.6f}")
    print(f"  Error:        {error:.6f}  ({error_pct:.4f}%)")
    if elapsed is not None:
        print(f"  Time:         {elapsed*1000:.1f} ms")
    print(f"{'='*52}\n")

    milestones = {100, 1_000, 10_000, 100_000, 1_000_000}
    rows = [(n, est) for n, est in running_estimates if n in milestones]
    if rows:
        print("  Convergence snapshots:")
        print(f"  {'Samples':>12}  {'π estimate':>12}  {'Error':>10}")
        print(f"  {'-'*38}")
        for n, est in rows:
            print(f"  {n:>12,}  {est:>12.6f}  {abs(est - math.pi):>10.6f}")
        print()


def benchmark(num_samples: int, seed: int = None):
    """Head-to-head speed comparison: pure Python vs NumPy."""
    print(f"\n{'='*52}")
    print(f"  Benchmark  (n={num_samples:,})")
    print(f"{'='*52}")

    t0 = time.perf_counter()
    pi_pure, _ = estimate_pi_pure(num_samples, seed=seed)
    t_pure = time.perf_counter() - t0

    t0 = time.perf_counter()
    pi_np, _ = estimate_pi_numpy(num_samples, seed=seed)
    t_np = time.perf_counter() - t0

    speedup = t_pure / t_np if t_np > 0 else float("inf")

    print(f"  {'Method':<14} {'π estimate':>12}  {'Time':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Pure Python':<14} {pi_pure:>12.6f}  {t_pure*1000:>8.1f} ms")
    print(f"  {'NumPy':<14} {pi_np:>12.6f}  {t_np*1000:>8.1f} ms")
    print(f"  {'-'*40}")
    print(f"  NumPy speedup: {speedup:.1f}x faster")
    print(f"{'='*52}\n")


def run_multiple_trials(num_samples: int, num_trials: int = 10):
    """Run repeated NumPy trials to illustrate variance."""
    estimates = []
    for _ in range(num_trials):
        est, _ = estimate_pi_numpy(num_samples)
        estimates.append(est)

    arr = np.array(estimates)
    mean_est = float(arr.mean())
    std_dev = float(arr.std())

    print(f"\n{'='*52}")
    print(f"  {num_trials} Trials  (n={num_samples:,} each, NumPy)")
    print(f"{'='*52}")
    for i, est in enumerate(estimates, 1):
        err = abs(est - math.pi)
        print(f"  Trial {i:>2}: {est:.5f}  (err {err:.5f})")
    print(f"  {'─'*40}")
    print(f"  Mean:    {mean_est:.6f}")
    print(f"  Std dev: {std_dev:.6f}")
    print(f"  True π:  {math.pi:.6f}")
    print(f"{'='*52}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo π estimation")
    parser.add_argument("-n", "--samples", type=int, default=1_000_000,
                        help="Number of random samples (default: 1,000,000)")
    parser.add_argument("-t", "--trials", type=int, default=0,
                        help="Run N repeated trials to show variance")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare pure Python vs NumPy speed")
    parser.add_argument("--pure", action="store_true",
                        help="Use pure-Python implementation instead of NumPy")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.samples, seed=args.seed)
    elif args.trials > 1:
        run_multiple_trials(args.samples, args.trials)
    else:
        fn = estimate_pi_pure if args.pure else estimate_pi_numpy
        label = "pure Python" if args.pure else "NumPy"
        t0 = time.perf_counter()
        pi_est, running = fn(args.samples, seed=args.seed)
        elapsed = time.perf_counter() - t0
        print_results(args.samples, pi_est, running, elapsed=elapsed, label=label)
