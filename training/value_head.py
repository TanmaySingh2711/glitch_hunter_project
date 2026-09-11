"""The one-off critic reset applied when the QA phase is seeded from 6M."""
from __future__ import annotations

import logging

from stable_baselines3.common.base_class import BaseAlgorithm

log = logging.getLogger(__name__)


def reset_value_head(model: BaseAlgorithm) -> bool:
    """Reinitialises the critic's output layer, leaving the policy untouched.

    ═══════════════════════════════════════════════════════════════════════
    WHY THIS IS NECESSARY, AND WHY IT IS SAFE
    ═══════════════════════════════════════════════════════════════════════
    Calibration measured the problem exactly. The legacy reward's typical
    episode returned ~625 (median; the mean of ~1748 is skewed by level
    completions), and roughly 2,180 points of a successful episode came from
    terms that are monotone in max-x - precisely the terms the QA reward
    deletes. The QA reward has no equivalent, so no safe novelty weight
    brings the two scales together: closing the gap would need a weight two
    orders of magnitude above the point where a single substep outweighs an
    entire legacy episode.

    That leaves a critic which confidently predicts returns in the hundreds
    for a reward that now produces returns near zero. Two consequences:

      1. Every advantage is computed against a badly wrong baseline. SB3
         normalises advantages per minibatch, so the ABSOLUTE error largely
         washes out - but the RELATIVE ordering across states is still driven
         by a value function fitted to a different objective, so the policy
         gradient points somewhere meaningless.

      2. Far worse: the value loss is enormous, on the order of the squared
         error between ~625 and ~0. With vf_coef=0.5 that gradient flows back
         through the SHARED CNN feature extractor. max_grad_norm=0.5 bounds
         its size but not its direction, so the first updates would be
         dominated by value error tearing at exactly the convolutional
         features that encode the locomotion this retrofit is trying to
         preserve.

    Reinitialising the value head to predict ~0 removes both. What is touched:
    ONLY policy.value_net, a single Linear(features_dim -> 1). What is NOT
    touched: the CNN feature extractor, the shared MLP extractor, and
    policy.action_net - i.e. every parameter that encodes how to run, jump,
    build momentum and avoid enemies. The agent wakes up able to play exactly
    as well as it did, but with no opinion about what states are worth.

    Small weights and a zero bias rather than a full random reinit: it starts
    the critic at "everything is worth about nothing", which is much closer to
    the truth for the QA reward than anything it currently believes, and it
    lets the value warm-up fit upward from a neutral prior instead of
    unlearning a confident wrong one.
    """
    import torch

    head = getattr(model.policy, "value_net", None)
    if head is None:
        log.warning("[VALUE HEAD] policy has no value_net attribute - skipping "
                    "reset. Check the SB3 version before trusting the first "
                    "iterations.")
        return False

    with torch.no_grad():
        before = float(head.weight.abs().mean())
        torch.nn.init.orthogonal_(head.weight, gain=0.01)
        if head.bias is not None:
            head.bias.zero_()
        after = float(head.weight.abs().mean())

    # The optimizer carries Adam moment estimates fitted to the old value
    # scale. Leaving them attached to freshly initialised weights would apply
    # months of accumulated momentum to parameters that no longer mean what
    # those moments were measuring.
    cleared = 0
    optimizer = model.policy.optimizer
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param is head.weight or param is head.bias:
                optimizer.state.pop(param, None)
                cleared += 1

    log.info("[VALUE HEAD] Reinitialised the critic output layer "
             "(mean |w| %.4f -> %.4f, bias zeroed, %d optimizer states cleared).",
             before, after, cleared)
    log.info("[VALUE HEAD] The CNN features and the action head are UNTOUCHED - "
             "locomotion is preserved; only the value estimate is reset.")
    return True
