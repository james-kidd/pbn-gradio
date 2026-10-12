"""
Linear programming as a certificate layer over the greedy heuristics.

The siting plan picks sites greedily. Greedy is fast and explainable, and it is
also unbounded in how bad it might be — nothing in the plan says whether it
found something near-optimal or left half the available value on the table.
Linear programming answers that, and the answer is what makes the heuristic
trustworthy rather than merely plausible.

**The primal.** Choosing a subset of candidate parcels to maximise total
suitability subject to a budget is a 0-1 knapsack:

    maximise    Σ vᵢ xᵢ
    subject to  Σ cᵢ xᵢ ≤ B
                xᵢ ∈ {0, 1}

NP-hard as stated. Relaxing xᵢ ∈ {0,1} to 0 ≤ xᵢ ≤ 1 gives a linear program,
solvable exactly, and — because every integral solution is also a feasible
fractional one — its optimum is an **upper bound** on the integer optimum.
So `greedy / LP` is a *certified* lower bound on how close greedy came to the
best achievable. Not an estimate: a proof.

**The dual.** The budget constraint carries one dual variable λ, and by strong
duality λ is the **shadow price of the budget** — the marginal suitability
bought by one more dollar, exactly, at the optimum. That is a number a planner
can act on directly: it answers "should I raise the budget?" without re-running
anything.

**Complementary slackness** ties them together. At the optimum, parcel i is
selected only if its reduced cost vᵢ − λcᵢ ≥ 0, which says something concrete:
λ is precisely the value-per-dollar threshold a parcel must clear to be worth
building. Every parcel's reduced cost is reported for that reason — it explains
inclusion and exclusion in one number.

**Scope, stated honestly.** This certifies *selection* under independent costs.
It does not certify *routing*, because shared corridors couple the costs and
break the knapsack structure. Routing has its own benchmark — the MST over the
metric closure in `topology.py`. Two guarantees over two separable decisions is
the correct division; one number pretending to cover both would not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog


@dataclass
class SelectionCertificate:
    """The LP bound, the dual, and what they say about a heuristic's choice."""
    lp_bound: float                       # upper bound on the integer optimum
    heuristic_value: float                # what the greedy plan achieved
    shadow_price: float                   # dual: suitability per extra dollar
    fractional: np.ndarray                # the LP's relaxed solution
    reduced_costs: list[float] = field(default_factory=list)
    n_fractional: int = 0
    budget: float = 0.0
    spent: float = 0.0
    feasible: bool = True
    heuristic_feasible: bool = True
    note: str = ""

    @property
    def ratio(self) -> float | None:
        """Certified fraction of the achievable optimum that the heuristic got.

        None when the heuristic's own selection breaks the budget. There is no
        meaningful ratio in that case: the LP bounds what is achievable *within*
        the constraint, so comparing an infeasible solution against it would
        certify nothing and could report above 100%.
        """
        if not self.heuristic_feasible or self.lp_bound <= 0:
            return None
        return float(np.clip(self.heuristic_value / self.lp_bound, 0.0, 1.0))

    @property
    def gap(self) -> float:
        return max(0.0, self.lp_bound - self.heuristic_value)


def budgeted_selection(
    values: list[float] | np.ndarray,
    costs: list[float] | np.ndarray,
    budget: float,
    chosen: list[int] | None = None,
) -> SelectionCertificate:
    """Solve the LP relaxation and certify a heuristic's selection against it.

    `chosen` is the index set the heuristic picked; its total value is compared
    against the LP bound. Passing None certifies nothing and just returns the
    bound and the dual.
    """
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64)
    n = len(v)

    if n == 0:
        return SelectionCertificate(0.0, 0.0, 0.0, np.array([]),
                                    note="no candidates")

    # scipy minimises, so the maximisation is passed as -v.
    result = linprog(
        -v,
        A_ub=c.reshape(1, -1),
        b_ub=[float(budget)],
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )

    if not result.success:
        return SelectionCertificate(
            float(v.sum()), 0.0, 0.0, np.zeros(n), feasible=False,
            note=f"LP did not solve: {result.message}",
        )

    x = np.asarray(result.x, dtype=np.float64)
    lp_bound = float(-result.fun)

    # The marginal is for the minimisation form; negate for the maximisation's
    # shadow price. It is non-negative: more budget never hurts.
    shadow = float(-result.ineqlin.marginals[0]) if result.ineqlin.marginals.size else 0.0

    # Complementary slackness: vᵢ − λcᵢ is the value-per-dollar margin over the
    # threshold λ. Positive means worth building, negative means not.
    reduced = (v - shadow * c).tolist()

    # A knapsack LP has at most one fractional variable at a basic optimum.
    # Checking it is a cheap confirmation that the solve is structurally sane.
    n_frac = int(np.sum((x > 1e-9) & (x < 1 - 1e-9)))

    heuristic_value = float(v[chosen].sum()) if chosen else 0.0
    spent = float(c[chosen].sum()) if chosen else 0.0

    # A solution that breaks the budget cannot be certified against a bound on
    # what the budget permits. Relative tolerance, since these are sums of
    # floats that were themselves derived from path integrals.
    heuristic_feasible = spent <= float(budget) * (1 + 1e-9) + 1e-6

    note = ""
    if not heuristic_feasible:
        note = (
            f"The selection costs ${spent:,.2f} against a budget of "
            f"${float(budget):,.2f}, so it is infeasible and no optimality "
            "ratio is claimed."
        )
    elif n_frac > 1:
        note = (
            f"{n_frac} fractional variables — a knapsack LP should have at most "
            "one at a basic optimum; the solve may be degenerate."
        )

    return SelectionCertificate(
        lp_bound=lp_bound,
        heuristic_value=heuristic_value,
        shadow_price=shadow,
        fractional=x,
        reduced_costs=reduced,
        n_fractional=n_frac,
        budget=float(budget),
        spent=spent,
        heuristic_feasible=heuristic_feasible,
        note=note,
    )


def format_certificate(
    cert: SelectionCertificate,
    labels: list[int] | None = None,
    chosen: list[int] | None = None,
) -> str:
    """The bound, the shadow price, and what they mean — as markdown."""
    if not cert.feasible:
        return f"### Selection certificate\n\nLP unavailable: {cert.note}"
    if cert.lp_bound <= 0:
        return ""

    lines = [
        "### Selection certificate (LP relaxation)",
        "",
        "Choosing parcels to maximise suitability under a budget is a 0-1 "
        "knapsack — NP-hard. Its LP relaxation is solvable exactly, and since "
        "every integral solution is also a feasible fractional one, the LP "
        "optimum **upper-bounds** the best achievable. That turns the greedy "
        "plan's quality from an assertion into a proof.",
        "",
        f"| | |",
        f"|---|---:|",
        f"| LP upper bound | **{cert.lp_bound:.2f}** |",
        f"| Greedy plan achieved | **{cert.heuristic_value:.2f}** |",
        (
            f"| Certified optimality | **≥ {100 * cert.ratio:.1f}%** |"
            if cert.ratio is not None
            else "| Certified optimality | **not claimed — selection over budget** |"
        ),
        f"| Value left on the table | at most {cert.gap:.2f} |",
        f"| Budget | ${cert.budget:,.0f} (spent ${cert.spent:,.0f}) |",
        "",
    ]

    if cert.shadow_price > 0:
        lines += [
            f"**Shadow price of the budget: {cert.shadow_price:.6g} suitability "
            "per dollar.**",
            "",
            "This is the dual variable on the budget constraint. By strong "
            "duality it is the exact marginal value of relaxing the budget at "
            "the optimum — so it answers *\"is another dollar worth spending?\"* "
            "without re-solving anything. Equivalently, by complementary "
            "slackness, it is the value-per-dollar threshold a parcel must "
            "clear to belong in the plan at all.",
        ]
    else:
        lines += [
            "**Shadow price is zero** — the budget is not binding. Everything "
            "worth building is affordable, so more money buys nothing. The "
            "constraint to relax is elsewhere.",
        ]

    if labels and cert.reduced_costs:
        ranked = sorted(
            zip(labels, cert.reduced_costs, cert.fractional),
            key=lambda t: t[1], reverse=True,
        )
        chosen_set = set(chosen or [])
        chosen_labels = {labels[i] for i in chosen_set if i < len(labels)}
        lines += [
            "",
            "Reduced cost **vᵢ − λcᵢ** per parcel — positive clears the "
            "threshold, negative does not:",
            "",
            "| Parcel | Reduced cost | LP | In plan |",
            "|---|---:|---:|:--:|",
        ]
        for lab, rc, xi in ranked[:8]:
            lines.append(
                f"| {lab} | {rc:+.3f} | {xi:.2f} | "
                f"{'✅' if lab in chosen_labels else '—'} |"
            )

    if cert.note:
        lines += ["", f"*{cert.note}*"]

    lines += [
        "",
        "*Scope: this certifies **selection** under independent costs. It does "
        "not certify **routing** — shared corridors couple the costs and break "
        "the knapsack structure. Routing is benchmarked separately against the "
        "MST over the metric closure.*",
    ]
    return "\n".join(lines)
