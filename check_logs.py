import glob, collections
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dirs = sorted(set(glob.glob("logs/PPO_28*") + glob.glob("logs/PPO_28")))
print("Found:", log_dirs)

for d in log_dirs:
    ea = EventAccumulator(d)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"{d}: no scalar data"); continue

    print(f"\n=== {d} ===")
    data = {}
    for tag in tags:
        evs = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in evs]

    steps = [s for s,_ in list(data.values())[0]]
    print(f"Steps: {steps[0]:,} -> {steps[-1]:,}  ({len(steps)} points)")

    key_metrics = [
        'rollout/ep_rew_mean', 'rollout/ep_len_mean',
        'train/entropy_loss', 'train/explained_variance',
        'train/approx_kl', 'train/policy_gradient_loss', 'train/value_loss'
    ]
    for m in key_metrics:
        if m not in data: continue
        pairs = data[m]
        first_s, first_v = pairs[0]
        last_s,  last_v  = pairs[-1]
        print(f"\n  {m}:")
        print(f"    first: {first_v:.4f} @ step {first_s:,}")
        print(f"    last : {last_v:.4f} @ step {last_s:,}")
        if 'rew' in m:
            peak = max(pairs, key=lambda x: x[1])
            print(f"    peak : {peak[1]:.4f} @ step {peak[0]:,}")
        if 'entropy' in m:
            trough = min(pairs, key=lambda x: x[1])
            print(f"    trough:{trough[1]:.4f} @ step {trough[0]:,}")
