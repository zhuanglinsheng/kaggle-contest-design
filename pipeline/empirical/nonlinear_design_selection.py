"""Deterministic selection from quality-validated original-game candidates.

An equal-cost fully revealing design takes priority over a finite-feedback
design. Such a tie supplies no cost advantage from withholding information.
This ranks evaluated candidates, without certifying a continuous optimum.
"""
from __future__ import annotations

import json
import math


def key(design):
    return json.dumps(design.record(), sort_keys=True)


def feasible_candidate_can_change_choices(contest_id, observed, candidate, candidates, validation, deltas):
    """Test cost/tie dominance without making any quality inference.

    If even declaring this candidate feasible cannot change a selected policy
    or a restricted full-disclosure comparator, solving its quality cannot
    affect the reported comparisons. Other prizes on its policy remain open.
    """
    candidates = set(candidates) | {candidate}
    actual = select_designs(contest_id, observed, candidates, validation, deltas)
    hypothetical = dict(validation)
    hypothetical[key(candidate)] = dict(passed=True)
    possible = select_designs(contest_id, observed, candidates, hypothetical, deltas)
    return any(a['selected'] != b['selected'] or a['full_comparator'] != b['full_comparator']
               for a, b in zip(actual, possible))


def select_designs(contest_id, observed, candidates, validation, deltas):
    candidates = set(candidates) | {observed}
    eligible = [d for d in candidates if d == observed or validation.get(key(d), {}).get('passed')]

    def stable_key(design):
        category = (0 if design.prize == 0 else 1 if design.precision == math.inf
                    else 2 if design.precision is None else 3)
        return category, design.horizon, design.prize, key(design)

    rows = []
    for delta in deltas:
        if not math.isfinite(delta) or delta < 0:
            raise ValueError('Finite nonnegative delay-cost coefficient required')

        def cost(design):
            return design.prize / observed.prize + delta * (design.horizon / observed.horizon)**2

        def choose(pool):
            if not pool:
                return None
            minimum = min(cost(d) for d in pool)
            tied = [d for d in pool if abs(cost(d)-minimum) <= 1e-12 * max(1., abs(minimum))]
            return min(tied, key=stable_key)

        selected = choose(eligible)
        full = choose([d for d in eligible if d.prize == 0 or d.precision == math.inf])
        kind = ('zero_prize' if selected.prize == 0 else 'full_disclosure'
                if selected.precision == math.inf else 'less_disclosure')
        additional = None
        if selected.prize > 0 and kind == 'less_disclosure' and full is not None:
            additional = 100 * (1-cost(selected)/cost(full))
        rows.append(dict(contest_id=contest_id, delta=delta, choice=kind,
                         prize_ratio=selected.prize/observed.prize,
                         time_multiplier=selected.horizon/observed.horizon,
                         cost_saving_pct=100*(1-cost(selected)/(1+delta)),
                         additional_reduction_pct=additional,
                         selected=selected.record(),
                         full_comparator=None if full is None else full.record(),
                         retained_observed=selected == observed))
    return rows
