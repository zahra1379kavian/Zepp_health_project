"""Fit the fused expert on training participants and the gate on calibration participants."""

import copy
import json

import numpy as np
import torch

from ...constants import FEATURES
from ...data import load_arrays, runs, write_json
from ...runner import digest, probabilities
from .model import PhaseHead, expert_features, gate_inputs, motion


def samples(base, wrist_root, paired_root, streams, rng, expert=None, cache=None):
    rows = []
    for i, name in enumerate(streams, 1):
        wrist = load_arrays(wrist_root / name)
        paired = load_arrays(paired_root / name)
        if not np.array_equal(wrist["t"], paired["t"]):
            raise ValueError(f"Misaligned prepared streams for {name}")
        if cache is not None and name in cache:
            q = cache[name]
        else:
            _, q = probabilities(base.model, wrist["x"][:, base.columns], wrist["valid"], base.norm, base.device)
            if cache is not None:
                cache[name] = q
        for a, b in runs(paired["valid"] & wrist["valid"]):
            # Uniform time subsample only the supervised rows. All causal features
            # and both branch predictions still see the original 25 Hz history.
            take = np.flatnonzero(paired["phase_mask"][a:b])
            take = take[rng.integers(min(8, len(take)))::8] if len(take) else take
            if not len(take):
                continue
            w, z, p = wrist["x"][a:b], paired["x"][a:b, 6:], q[a:b]
            if expert is None:
                x = expert_features(w, z, p, motion(w, z))
                inputs = (x, p)
            else:
                x, fused = gate_inputs(expert, w, z, p)
                inputs = (x, p, fused)
            rows.append(tuple(v[take] for v in (*inputs, paired["phase"][a:b])))
        if i % 5 == 0 or i == len(streams):
            print(f"Prepared {i}/{len(streams)} labeled wrist streams", flush=True)
    if not rows:
        raise ValueError("No aligned, labeled samples in partition")
    return tuple(torch.from_numpy(np.concatenate(column)) for column in zip(*rows))


def fit_head(head, train, validation, epochs=20):
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    counts = train[-1].bincount(minlength=3)
    if (counts == 0).any():
        raise ValueError("A phase class has no training samples")
    class_weights = counts.float().rsqrt()
    head.mean.copy_(train[0].mean(0))
    head.scale.copy_(train[0].std(0, unbiased=False).clamp_min(.001))

    def loss(inputs, targets):
        q, g = head(*inputs)
        nll = -q.gather(1, targets[:, None]).clamp_min(1e-7).log().flatten()
        weights = class_weights[targets]
        value = (nll * weights).sum() / weights.sum()
        return value if g is None else value + .01 * g.mean()

    best, history = None, []
    for epoch in range(1, epochs + 1):
        for idx in torch.randperm(len(train[-1])).split(2048):
            value = loss([v[idx] for v in train[:-1]], train[-1][idx])
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
        with torch.no_grad():
            val = float(loss(validation[:-1], validation[-1]))
        if not np.isfinite(val):
            raise ValueError("Nonfinite phase loss")
        history.append(val)
        print(f"{'Gate' if head.gating else 'Fused expert'} epoch {epoch}: validation loss {val:.5f}",
              flush=True)
        if best is None or val < best[0]:
            best = (val, epoch, copy.deepcopy(head.state_dict()))
        if epoch - best[1] >= 4:
            break
    head.load_state_dict(best[2])
    return dict(samples=len(train[-1]), class_counts=counts.tolist(), validation_samples=len(validation[-1]),
                best_epoch=best[1], validation_losses=history)


def fit(base, wrist_root, paired_root, parts, output, seed):
    paired = json.loads((paired_root / "manifest.json").read_text())
    wrist = json.loads((wrist_root / "manifest.json").read_text())
    if paired["split"] != wrist["split"] or paired["features"] != [
            f"{site}_{axis}" for site in ("wrist", "waist") for axis in FEATURES[:6]]:
        raise ValueError("Unexpected paired view or subject split")
    people = {r["session"]: r["subject"] for r in wrist["sessions"]}
    for part, streams in wrist["split"].items():
        if {people[s] for s in streams} != set(parts[part]):
            raise ValueError(f"Benchmark subject mismatch in {part}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    val_people = rng.permutation(sorted(parts["validation"])).tolist()
    selection = set(val_people[:max(1, len(val_people) // 3)])
    val_streams = wrist["split"]["validation"]
    calibration = [s for s in val_streams if people[s] not in selection]
    validation = [s for s in val_streams if people[s] in selection]
    plan = dict(seed=seed, expert_subjects=parts["train"], calibration_subjects=sorted(set(val_people)-selection),
                selection_subjects=sorted(selection), test_subjects=parts["test"], wrist=base.metadata,
                paired_manifest_sha256=digest(paired_root / "manifest.json"), paired_data=str(paired_root),
                model_lookahead_ms=0, decoder_confirmation_ms=base.config.confirmation_frames * 40,
                gate_use_penalty=.01, history_frames=[1, 5, 15, 40], supervision_stride=8)
    write_json(output / "fit.json", plan)
    expert, gate = PhaseHead(False), PhaseHead(True)
    cache = {}
    print("Preparing fused expert training data", flush=True)
    train = samples(base, wrist_root, paired_root, wrist["split"]["train"], rng)
    val = samples(base, wrist_root, paired_root, validation, rng, cache=cache)
    plan["expert"] = fit_head(expert, train, val)
    del train, val
    print("Preparing gate calibration data", flush=True)
    train = samples(base, wrist_root, paired_root, calibration, rng, expert)
    val = samples(base, wrist_root, paired_root, validation, rng, expert, cache)
    plan["gate"] = fit_head(gate, train, val)
    torch.save(dict(expert=expert.state_dict(), gate=gate.state_dict(), wrist=base.metadata), output / "heads.pt")
    write_json(output / "fit.json", plan)
    return expert.eval(), gate.eval()
