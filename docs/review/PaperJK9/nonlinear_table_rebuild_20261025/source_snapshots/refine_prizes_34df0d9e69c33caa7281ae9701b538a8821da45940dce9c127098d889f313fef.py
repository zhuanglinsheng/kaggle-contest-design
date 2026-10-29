"""Refine completed original-model incumbents without changing the paper.

Passing means the existing paired numerical diagnostic supports feasibility.
Failure to pass does NOT establish infeasibility. A local crossing search is
not a proof of monotonicity or of a global minimum. Work is limited to the
designs and comparators needed for the existing manuscript table.
"""
from pathlib import Path
import argparse
import ast
import concurrent.futures as futures
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nonlinear_batch", HERE / "design_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
from pipeline.empirical.nonlinear_design_recovery import recover
from pipeline.empirical.nonlinear_design_selection import select_designs, feasible_candidate_can_change_choices


class RefinedCase(batch.Case):
    def __init__(self, cid, accepted):
        super().__init__(cid, accepted)
        recovery_source = Path(batch.root / 'pipeline/empirical/nonlinear_design_recovery.py')
        class_node = next(n for n in ast.parse(Path(__file__).read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == 'RefinedCase')
        fine_node = next(n for n in class_node.body
                         if isinstance(n, ast.FunctionDef) and n.name == 'fine_output')
        identity = dict(base=self.provenance['validation_protocol_sha256'],
                        fine_output=ast.dump(fine_node, include_attributes=False),
                        recovery=hashlib.sha256(recovery_source.read_bytes()).hexdigest())
        self.provenance['validation_protocol_sha256'] = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()
        progress = self.folder / 'validation_progress.json'
        if progress.exists():
            stored = json.loads(progress.read_text())
            if (stored.get('protocol_sha256') == self.provenance['validation_protocol_sha256']
                    and stored.get('accepted_sha256') == self.provenance['accepted_sha256']):
                self.validated.update(stored['results'])

    def fine_output(self, draw, design):
        result = super().fine_output(draw, design)
        if result['resolved']:
            return result
        recovered = recover(draw, design, cache_dir=self.folder / 'cross_coordinate_recovery',
                            build_dir=HERE / 'build')
        return dict(recovered, previous_value_HJB=result)

    def validate(self, design):
        # Unknown is not an infeasibility result. Underlying numerical caches
        # may now contain an independently refined baseline or recovered point.
        # Retry only when this candidate is actually requested by the table's
        # prize/comparator search; keep every unrequested unknown for reporting.
        key = json.dumps(design.record(), sort_keys=True)
        if self.validated.get(key, {}).get('status') == 'unresolved':
            self.validated.pop(key)
        return super().validate(design)


def refine_policy(prize, evaluate, *, scale, relative_tolerance=1e-4, steps=40,
                  initial_trials=None, upper_prize=None):
    """Sample a local lower supported crossing, retaining all uncertain probes.

    The coarse geometric probes can discover separated feasible components;
    unsampled intervals remain unexcluded. No claim of a global root is made.
    """
    if prize < 0 or scale <= 0:
        raise ValueError("Nonnegative incumbent and positive prize scale required")
    trials = {} if initial_trials is None else {float(k): v for k, v in initial_trials.items()}

    def check(value):
        value = float(value)
        if value not in trials:
            trials[value] = evaluate(value)
        return bool(trials[value].get("passed"))

    check(prize)
    check(0.)
    if trials[0.].get("passed"):
        return dict(prize=0., trials=trials, local_width=0., supported=True,
                    scope="Zero-prize feasible diagnostic; no global design claim")
    # A resumed policy often already has a narrow tested bracket. Avoid
    # reevaluating geometric probes that cannot improve that bracket.
    if not any(0 < p < prize for p in trials):
        for ratio in (.125, .25, .5):
            check(prize * ratio)
    supported = [p for p in trials if trials[p].get("passed")]
    # Full-disclosure policies may need a larger prize than the initial
    # screening grid proposed. Search upward only while the prize could
    # improve a currently verified restricted comparator's cost.
    if not supported and upper_prize is not None:
        upper_prize = max(0., float(upper_prize))
        current = min(max(prize, .125*scale), upper_prize)
        while current > 0:
            if check(current):
                break
            if current >= upper_prize:
                break
            current = min(2*current, upper_prize)
        supported = [p for p in trials if trials[p].get('passed')]
    if not supported:
        return dict(prize=None, trials=trials, supported=False,
                    scope="No supported prize found among evaluated probes")
    high = min(supported)
    low = max(p for p in trials if p < high)
    for iteration in range(steps):
        if high - low <= relative_tolerance * scale:
            break
        mid = (low + high) / 2
        flo = trials[low].get('lower_diagnostic_margin')
        fhi = trials[high].get('lower_diagnostic_margin')
        if trials[high].get('status') == 'identity':
            fhi = 0.  # Identical original policy: paired difference is exactly zero.
        # Interpolation proposes a price; only a fresh original-model
        # evaluation decides acceptance. Periodic bisection safeguards the
        # local search without assuming global prize monotonicity.
        if (iteration % 3 != 2 and flo is not None and fhi is not None
                and math.isfinite(flo) and math.isfinite(fhi) and flo < 0 <= fhi):
            root = low - flo * (high-low) / (fhi-flo)
            padding = .25 * relative_tolerance * scale
            proposed = root + (padding if iteration % 2 == 0 else -padding)
            if proposed >= high:
                proposed = root-padding
            elif proposed <= low:
                proposed = root+padding
            if low < proposed < high and proposed not in trials:
                mid = proposed
        if check(mid):
            high = mid
        else:
            low = mid
    return dict(prize=high, trials=trials, local_width=high-low,
                supported=True, scope="Lowest supported sampled local crossing; "
                "nonpassing and unsampled prizes are not excluded as infeasible")


def refine_case(path):
    original = json.loads(path.read_text())
    cid = original["contest_id"]
    case = RefinedCase(cid, Path(original["provenance"]["accepted"]))
    original_prize_probes = []
    for ratio in [.125, .25, .5, .75, .9]:
        candidate = batch.Design(case.observed.prize*ratio,case.observed.horizon,None)
        original_prize_probes.append(dict(ratio=ratio,design=candidate.record(),
                                         validation=case.validate(candidate)))
    unresolved = [k for k, v in case.validated.items() if v.get('status') == 'unresolved']
    # Do not turn table recomputation into recovery of the entire failed
    # candidate grid. Keep these candidates unknown. Further recovery is
    # performed only if evaluating a required policy reaches a new point.
    deferred_unresolved = [json.loads(key) for key in unresolved]
    # A completed first pass no longer writes this folder. Use its complete
    # validation cache, including unsuccessful and unresolved candidates.
    policies = {}
    search = json.loads((case.folder / "search_progress.json").read_text())
    for record in search["proposals"]:
        if record["precision"] == "full":
            key = (record["horizon"], "full")
            policies[key] = max(policies.get(key, 0.), record["prize"])
    # Treat the selected finite policies with exactly the same prize procedure.
    for selection in original["selections"]:
        record = selection["selected"]
        precision = record["precision"]
        key = (record["horizon"], precision)
        policies[key] = max(policies.get(key, 0.), record["prize"])
    for multiple in batch.TIMES:
        policies.setdefault((case.observed.horizon * multiple, "full"), case.observed.prize)
    policies.setdefault((case.observed.horizon,'observed'),case.observed.prize)
    def current_selections():
        designs = []
        for encoded in case.validated:
            record = json.loads(encoded)
            precision = record['precision']
            designs.append(batch.Design(record['prize'], record['horizon'],
                math.inf if precision == 'full' else None if precision == 'observed' else precision))
        return select_designs(cid, case.observed, designs, case.validated, batch.DELTAS)
    for selection in current_selections():
        record = selection['selected']
        policies.setdefault((record['horizon'], record['precision']), record['prize'])
    for key, value in list(policies.items()):
        passed = [json.loads(k)["prize"] for k, v in case.validated.items()
                  if v.get("passed") and
                  (json.loads(k)["horizon"], json.loads(k)["precision"]) == key]
        if key == (case.observed.horizon, "observed"):
            passed.append(case.observed.prize)
        if passed:
            policies[key] = min(passed)
    dest = case.folder / "prize_refinement.json"
    result = dict(contest_id=cid, status="running", publication_ready=False,
                  original_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  refinement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policies=[], newly_selected_policies=[],
                  deferred_unresolved_candidates=deferred_unresolved,
                  original_prize_probe_version=1,original_prize_probes=original_prize_probes,
                  provenance=case.provenance,
                  remaining="Existing-table prize/full-disclosure comparisons and final table arithmetic; "
                  "unresolved unselected candidates remain explicitly delimited")
    batch.save(dest, result)
    pending = sorted(policies.items(), key=lambda item: str(item[0]))
    processed = set()
    while pending:
        (horizon, precision), prize = pending.pop(0)
        processed.add((horizon, precision))
        decoded = math.inf if precision == "full" else None if precision == "observed" else precision

        zero_result = next((value for encoded, value in case.validated.items()
                            if value.get('passed')
                            and json.loads(encoded)['prize'] == 0
                            and json.loads(encoded)['horizon'] == horizon), None)
        if zero_result is not None:
            result['policies'].append(dict(horizon=horizon, precision=precision,
                supported=True, prize=0., local_width=0., trials={0.: zero_result},
                scope='Already verified zero prize; feedback is irrelevant'))
            batch.save(dest, result)
            continue

        # Even a free prize cannot improve a required comparison at this
        # duration: all positive-prize policies here are cost dominated.
        candidates = []
        for encoded in case.validated:
            record = json.loads(encoded)
            value = record['precision']
            candidates.append(batch.Design(record['prize'], record['horizon'],
                math.inf if value == 'full' else None if value == 'observed' else value))
        free = batch.Design(0., horizon, None)
        if not feasible_candidate_can_change_choices(cid, case.observed, free,
                candidates, case.validated, batch.DELTAS):
            result['policies'].append(dict(horizon=horizon, precision=precision,
                supported=False, prize=None, cost_dominated=True,
                scope='Even zero prize cannot improve a required table comparison'))
            batch.save(dest, result)
            continue

        def evaluate(value):
            return case.validate(batch.Design(value, horizon, decoded))

        known = {json.loads(k)['prize']: v for k, v in case.validated.items()
                 if v.get('status') != 'unresolved' and
                 (json.loads(k)['horizon'], json.loads(k)['precision']) == (horizon, precision)}
        if (horizon, precision) == (case.observed.horizon, 'observed'):
            known[case.observed.prize] = dict(passed=True, status='identity')
        upper = None
        if precision == 'full':
            full_candidates = []
            for encoded, value in case.validated.items():
                record = json.loads(encoded)
                if value.get('passed') and (record['prize'] == 0 or record['precision'] == 'full'):
                    full_candidates.append(batch.Design(record['prize'], record['horizon'],
                        math.inf if record['precision'] == 'full' else None if record['precision'] == 'observed' else record['precision']))
            if full_candidates:
                upper = case.observed.prize * max(0., max(
                    min(case.cost(d, delta) for d in full_candidates)
                    - delta * (horizon/case.observed.horizon)**2 for delta in batch.DELTAS))
            else:
                upper = 128 * case.observed.prize
        refined = refine_policy(prize, evaluate, scale=case.observed.prize,
                                initial_trials=known, upper_prize=upper)
        result["policies"].append(dict(horizon=horizon, precision=precision,
                                      upward_search_cost_cap=upper, **refined))
        queued = {key for key, _ in pending}
        for selection in current_selections():
            record = selection['selected']
            policy = (record['horizon'], record['precision'])
            if policy not in processed and policy not in queued:
                pending.append((policy, record['prize']))
                queued.add(policy)
                result['newly_selected_policies'].append(record)
        batch.save(dest, result)
    result["status"] = "completed_local_prize_refinement"
    batch.save(dest, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contest", type=int)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument('--workers', type=int, choices=range(1,7), default=2)
    args = parser.parse_args()
    lock = (HERE / "refinement_controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status="watching_completed_first_pass", pid=os.getpid(), workers=args.workers,
                 completed={}, failed={}, active={}, publication_ready=False)
    status_path = HERE / "refinement_status.json"
    jobs = {}
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            ready = []
            for path in sorted(HERE.glob('contests/*/result.json'), key=lambda p: int(p.parent.name)):
                cid = path.parent.name
                if args.contest is not None and int(cid) != args.contest:
                    continue
                dest = path.parent/'prize_refinement.json'
                saved = json.loads(dest.read_text()) if dest.exists() else {}
                if saved.get('status') == 'completed_local_prize_refinement' and saved.get('original_prize_probe_version') == 1:
                    state['completed'][cid] = str(dest)
                    continue
                if cid not in state['failed'] and cid not in state['active']:
                    ready.append(path)
            while ready and len(jobs) < args.workers:
                path = ready.pop(0); cid = path.parent.name
                jobs[pool.submit(refine_case,path)] = cid
                state['active'][cid] = str(path)
            state['status'] = 'refining' if jobs else 'watching_completed_first_pass'
            batch.save(status_path,state)
            if jobs:
                done,_ = futures.wait(jobs,timeout=5,return_when=futures.FIRST_COMPLETED)
                for job in done:
                    cid = jobs.pop(job); state['active'].pop(cid,None)
                    try:
                        job.result()
                        state['completed'][cid] = str(HERE/f'contests/{cid}/prize_refinement.json')
                    except Exception as error:
                        state['failed'][cid] = repr(error)
                batch.save(status_path,state)
                continue
            status = json.loads((HERE/'design_status.json').read_text())
            if not args.watch or status['status'].startswith('completed'):
                break
            time.sleep(5)
    state["status"] = "completed_local_stage" if not state["failed"] else "local_stage_with_failures"
    batch.save(status_path, state)
