#!/usr/bin/env python3
"""Reference implementation of the head-agnostic speculative verify core.

This is the algorithmic heart of M3.2, written so it can be tested without a
GPU or a model.  The pinned engine (llama.cpp ``b11223``) already runs this loop
in ``common/speculative.cpp``; this module fixes the exact contract in one small
place, proves greedy equivalence, and documents what the engine must guarantee:

    prepare_verify_inputs(prefix, draft)
        -> target forward over all draft positions in one pass
        -> accept_longest_greedy_prefix(...)      (accept the longest common
           prefix with the target's own greedy tokens; the bonus token is the
           target token at the first mismatch)
        -> atomic prefix commit                    (proposal and KV advances are
           all-or-nothing, so a rejected draft never leaks into the context)

Drafter: a *point-mass* suffix/n-gram proposal.  It looks up the longest suffix
of the context that occurred earlier and proposes the tokens that followed that
earlier occurrence.  The proposal is deterministic, so it carries no sampling
noise and cannot change a temperature-0 result.

The engine's real numbers come from ``bench/measure-speculation.py``; this file
is the equivalence test and the reference for review.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Token = int


# ---------------------------------------------------------------------------
# point-mass suffix / n-gram drafter
# ---------------------------------------------------------------------------


class NgramDrafter:
    """A deterministic suffix/n-gram proposal over a token history.

    For a context, find the longest suffix of length ``min(order, len(context)-1)``
    down to ``1`` that occurred at least once earlier; propose up to ``m`` tokens
    that followed the most recent earlier occurrence.  No match -> empty draft.
    """

    def __init__(self, order: int = 4, max_draft: int = 8, min_order: int = 1):
        self.order = max(1, int(order))
        self.max_draft = max(1, int(max_draft))
        self.min_order = max(1, int(min_order))
        self.calls = 0
        self.empty = 0

    def propose(self, context: Sequence[Token]) -> List[Token]:
        self.calls += 1
        ctx = list(context)
        if len(ctx) < 2:
            self.empty += 1
            return []
        top = min(self.order, len(ctx) - 1)
        for n in range(top, self.min_order - 1, -1):
            needle = tuple(ctx[-n:])
            best: Optional[List[Token]] = None
            # Scan right-to-left so the most recent occurrence wins (recency).
            for i in range(len(ctx) - n - 1, -1, -1):
                if tuple(ctx[i:i + n]) == needle:
                    best = ctx[i + n:i + n + self.max_draft]
                    break
            if best:
                return best
        self.empty += 1
        return []


# ---------------------------------------------------------------------------
# verify core
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    accepted: List[Token] = field(default_factory=list)
    bonus: Optional[Token] = None
    rejected_at: Optional[int] = None


def target_forward(target_next: Callable[[Sequence[Token]], Token],
                   prefix: Sequence[Token],
                   draft: Sequence[Token]) -> List[Token]:
    """The greedy target token at every draft position (one conceptual pass).

    ``target_next(prompt + accepted_so_far)`` is the target model's greedy next
    token.  In the engine this is a single batched forward over the draft rows;
    here it is a sequence of deterministic calls, which is equivalent for a
    temperature-0 greedy verify.
    """
    base = list(prefix)
    return [target_next(base + list(draft[:i])) for i in range(len(draft))]


def accept_longest_greedy_prefix(target_tokens: Sequence[Token],
                                 draft: Sequence[Token]) -> VerifyResult:
    """Accept the longest prefix of ``draft`` that matches the target greedy path."""
    accepted: List[Token] = []
    for i, d in enumerate(draft):
        t = target_tokens[i]
        if t == d:
            accepted.append(d)
        else:
            return VerifyResult(accepted=accepted, bonus=t, rejected_at=i)
    bonus = target_tokens[len(draft)] if len(target_tokens) > len(draft) else None
    return VerifyResult(accepted=accepted, bonus=bonus, rejected_at=None)


def prepare_verify_inputs(prefix: Sequence[Token], draft: Sequence[Token]) -> List[Token]:
    """Inputs fed to the target forward: the prefix plus the whole draft.

    Returning the concatenation makes the atomicity requirement explicit: the
    committed context advances by ``accepted + [bonus]`` in one step, never by
    the raw draft.
    """
    return list(prefix) + list(draft)


# ---------------------------------------------------------------------------
# online acceptance policy (Strata-style window sizing)
# ---------------------------------------------------------------------------


class AcceptancePolicy:
    """Size the draft window from measured tokens-per-round, not a fixed window.

    ``acceptance`` is the fraction of proposed tokens the target accepts;
    ``tokens_per_round`` is ``1 + accepted`` per verify round.  The window grows
    while recent acceptance beats ``grow_at`` and shrinks toward 1 when it falls
    below ``shrink_at`` (so a low-acceptance workload degrades to no speculation
    rather than paying for drafts).
    """

    def __init__(self, n_min: int = 1, n_max: int = 8, grow_at: float = 0.6,
                 shrink_at: float = 0.3, alpha: float = 0.25):
        self.n_min = max(1, n_min)
        self.n_max = max(self.n_min, n_max)
        self.grow_at = grow_at
        self.shrink_at = shrink_at
        self.alpha = alpha
        self.acceptance_ewma: Optional[float] = None
        self.window = self.n_min

    def observe(self, accepted: int, drafted: int) -> None:
        if drafted <= 0:
            return
        ratio = accepted / drafted
        self.acceptance_ewma = ratio if self.acceptance_ewma is None else (
            self.alpha * ratio + (1 - self.alpha) * self.acceptance_ewma
        )
        if self.acceptance_ewma >= self.grow_at and self.window < self.n_max:
            self.window += 1
        elif self.acceptance_ewma < self.shrink_at:
            self.window = self.n_min


@dataclass
class SpecStats:
    drafted: int = 0
    accepted: int = 0
    rounds: int = 0
    tokens: int = 0

    @property
    def acceptance(self) -> Optional[float]:
        return (self.accepted / self.drafted) if self.drafted else None

    @property
    def tokens_per_round(self) -> Optional[float]:
        return (self.tokens / self.rounds) if self.rounds else None


def speculative_greedy_decode(prompt: Sequence[Token],
                              target_next: Callable[[Sequence[Token]], Token],
                              drafter: NgramDrafter,
                              max_new: int,
                              policy: Optional[AcceptancePolicy] = None) -> Tuple[List[Token], SpecStats]:
    """Greedy decode with point-mass n-gram drafts and exact verification."""
    tokens: List[Token] = list(prompt)
    stats = SpecStats()
    while len(tokens) - len(prompt) < max_new:
        want = min(policy.window if policy else drafter.max_draft, max_new - (len(tokens) - len(prompt)))
        draft = drafter.propose(tokens)[:want]
        target_tokens = target_forward(target_next, tokens, draft)
        vr = accept_longest_greedy_prefix(target_tokens, draft)
        committed = vr.accepted + ([] if vr.bonus is None else [vr.bonus])
        if not committed:
            # Degenerate target (should not happen for greedy); take one token.
            committed = [target_next(tokens)]
        tokens.extend(committed)
        stats.drafted += len(draft)
        stats.accepted += len(vr.accepted)
        stats.rounds += 1
        stats.tokens += len(committed)
        if policy:
            policy.observe(len(vr.accepted), len(draft) if draft else 1)
    return tokens[:len(prompt) + max_new], stats


def plain_greedy_decode(prompt: Sequence[Token],
                        target_next: Callable[[Sequence[Token]], Token],
                        max_new: int) -> List[Token]:
    tokens = list(prompt)
    for _ in range(max_new):
        tokens.append(target_next(tokens))
    return tokens


# ---------------------------------------------------------------------------
# deterministic synthetic target for the tests
# ---------------------------------------------------------------------------


class MarkovTarget:
    """A deterministic order-k Markov target built from a seeded token sample.

    Because the target is exactly the kind of repeatable process an n-gram
    drafter can learn, it exercises both high-acceptance and low-acceptance
    regimes.
    """

    def __init__(self, order: int = 4, vocab: int = 16, seed: int = 7):
        self.order = order
        self.vocab = vocab
        self.table: Dict[Tuple[int, ...], int] = {}
        # A fixed cycle guarantees long repeatable runs; the hash adds variety
        # so low-order drafters are not trivially perfect.
        h = hashlib.sha256(str(seed).encode()).digest()
        self.cycle = [h[i % len(h)] % vocab for i in range(32)]

    def next_token(self, prefix: Sequence[Token]) -> int:
        pos = len(prefix)
        key = tuple(prefix[-self.order:]) if self.order else ()
        if key in self.table:
            return self.table[key]
        digest = hashlib.sha256(repr(key).encode()).digest()
        # 50%: follow the periodic cycle; 50%: deterministic hash choice.  The
        # periodic half is what a suffix drafter reproduces.
        if digest[0] % 2 == 0:
            tok = self.cycle[pos % len(self.cycle)]
        else:
            tok = digest[1] % self.vocab
        self.table[key] = tok
        return tok


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def _run_equivalence(order: int, vocab: int, prompt_len: int, max_new: int, seed: int) -> Tuple[bool, SpecStats]:
    target = MarkovTarget(order=order, vocab=vocab, seed=seed)
    drafter = NgramDrafter(order=order, max_draft=6)
    policy = AcceptancePolicy(n_min=1, n_max=6)
    prompt = [hashlib.sha256(f"{seed}:{i}".encode()).digest()[0] % vocab for i in range(prompt_len)]
    spec, stats = speculative_greedy_decode(prompt, target.next_token, drafter, max_new, policy)
    plain = plain_greedy_decode(prompt, target.next_token, max_new)
    return spec == plain, stats


def self_test() -> int:
    # 1. Equivalence: speculative == plain greedy across model orders and seeds.
    total = 0
    for order in (1, 2, 3, 4, 6):
        for seed in range(8):
            same, stats = _run_equivalence(order, 16, 64, 96, seed)
            assert same, f"divergence: order={order} seed={seed}"
            total += stats.rounds
    print(f"equivalence: OK ({5 * 8} traces, {total} verify rounds)")

    # 2. The verify core never commits a rejected draft token.
    target = MarkovTarget(order=2, vocab=8, seed=3)
    prefix = [1, 2, 3, 4]
    draft = [target.next_token(prefix), 999, 998]
    tt = target_forward(target.next_token, prefix, draft)
    vr = accept_longest_greedy_prefix(tt, draft)
    assert vr.rejected_at in (0, 1, 2)
    if vr.rejected_at is not None:
        assert vr.accepted == draft[:vr.rejected_at], (vr, draft)
        assert vr.bonus == tt[vr.rejected_at]
    print("verify core: rejects at the first mismatch: OK")

    # 3. Window shrinks to 1 when acceptance is poor, grows when it is good.
    pol = AcceptancePolicy(n_min=1, n_max=4, grow_at=0.6, shrink_at=0.3)
    for _ in range(20):
        pol.observe(0, 4)
    assert pol.window == 1, pol.window
    for _ in range(20):
        pol.observe(4, 4)
    assert pol.window == 4, pol.window
    print("acceptance policy: shrinks low-acceptance, grows high-acceptance: OK")

    # 4. A perfect drafter reaches >1 accepted token per round.
    class CopyDrafter(NgramDrafter):
        def propose(self, context):
            return [context[-1]] * self.max_draft

    # target that repeats the last token -> copy drafter is always right
    repeat_target = lambda prefix: prefix[-1]
    _, stats = speculative_greedy_decode([5], repeat_target, CopyDrafter(order=1, max_draft=6), 60)
    assert stats.tokens_per_round and stats.tokens_per_round > 1.5, stats
    print(f"speculation: perfect drafter tokens/round={stats.tokens_per_round:.2f}: OK")

    # 5. A drafter that never matches cannot regress correctness (window -> 1).
    class NullDrafter(NgramDrafter):
        def propose(self, context):
            return []

    target2 = MarkovTarget(order=3, vocab=12, seed=11)
    prompt = [1, 2, 3, 4, 5]
    spec, stats = speculative_greedy_decode(prompt, target2.next_token, NullDrafter(), 40)
    plain = plain_greedy_decode(prompt, target2.next_token, 40)
    assert spec == plain, "empty-draft path changed output"
    assert stats.acceptance is None or stats.acceptance == 0
    print("no-draft fallback: output identical, no regression: OK")

    print("spec-verify-core self-test: ALL OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="reference speculative verify core")
    p.add_argument("command", nargs="?", default="self-test", choices=["self-test"])
    p.parse_args(argv)
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
