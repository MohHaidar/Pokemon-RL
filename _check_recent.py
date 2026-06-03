from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for run in ["PPO_24", "PPO_25", "PPO_26"]:
    d = f"logs/{run}"
    ea = EventAccumulator(d); ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"{run}: no data"); continue
    data = {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in tags}
    steps = [s for s, _ in list(data.values())[0]]
    print(f"\n=== {run} === {steps[0]:,} -> {steps[-1]:,} steps ({len(steps)} pts)")
    for m in ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/entropy_loss', 'train/explained_variance']:
        if m not in data: continue
        pairs = data[m]
        fv = pairs[0][1]; lv = pairs[-1][1]
        line = f"  {m}: {fv:.3f} -> {lv:.3f}"
        if 'rew' in m:
            pk = max(pairs, key=lambda x: x[1])
            line += f"  (peak {pk[1]:.3f} @ {pk[0]:,})"
        print(line)
