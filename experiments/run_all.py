"""Run all experiments sequentially and print timing."""
import time
import sys

experiments = [
    ('Capacity & Ablation', 'capacity_test'),
    ('Robustness', 'robustness_test'),
    ('Mechanistic Analysis', 'mechanistic_analysis'),
    ('Detectability', 'detectability_test'),
]

if __name__ == '__main__':
    t_total = time.time()

    for name, module in experiments:
        print(f"\n{'#' * 70}")
        print(f"# EXPERIMENT: {name}")
        print(f"{'#' * 70}")
        t0 = time.time()
        try:
            mod = __import__(module)
            mod.main()
            print(f"\n>>> {name} completed in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"\n>>> {name} FAILED after {time.time()-t0:.0f}s: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#' * 70}")
    print(f"# ALL EXPERIMENTS DONE in {time.time()-t_total:.0f}s")
    print(f"{'#' * 70}")
