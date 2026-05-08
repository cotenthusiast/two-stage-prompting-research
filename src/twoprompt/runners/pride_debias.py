# src/twoprompt/runners/pride_debias.py

"""PriDe (Zheng et al., ICLR 2024) — probabilities from token logprobs.

Implements cyclic permutation pooling (their Eq.~(1)), per-sample prior
estimation via Eq.~(7), averaging into a global prior, and Eq.~(8) debiasing
for held-out samples. Open-weight path: first-completion-token logprobs for
letters A/B/C/D (Together ``request_logprobs``), mapped to normalized label
distribution per §2.2 of the paper.

References:
    Chujie Zheng et al., "Large Language Models Are Not Robust Multiple Choice
    Selectors", ICLR 2024 (arXiv:2309.03882).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


def _normalize_option_letter(token: str | None) -> str | None:
    if token is None:
        return None
    stripped = token.strip().upper()
    return stripped if stripped in set(OPTION_LETTERS) else None


def merge_option_logprobs(logprobs: list[Any] | None) -> dict[str, float]:
    """Together-style first-token logprobs merged to max logprob per A–D."""
    if not logprobs:
        return {}
    first = logprobs[0]
    if not isinstance(first, Mapping):
        return {}

    tuples: list[tuple[str | None, Any]] = [
        (first.get("token"), first.get("logprob")),
    ]
    tops = first.get("top_logprobs") or []
    if isinstance(tops, list):
        for t in tops:
            if isinstance(t, Mapping):
                tuples.append((t.get("token"), t.get("logprob")))

    best: dict[str, float] = {}
    for tok, lp in tuples:
        letter = _normalize_option_letter(tok if isinstance(tok, str) else None)
        if letter is None or lp is None:
            continue
        try:
            lp_f = float(lp)
        except (TypeError, ValueError):
            continue
        prev = best.get(letter)
        if prev is None or lp_f > prev:
            best[letter] = lp_f
    return best


_LOGPROB_FLOOR = -30.0


def logprob_map_to_label_distribution(
        logp_map: Mapping[str, float],
        *,
        letters: Iterable[str] = OPTION_LETTERS,
        eps_prob: float = 1e-12,
) -> np.ndarray:
    """Softmax over option-ID letters from merged first-token logprobs.

    Missing letters receive ``_LOGPROB_FLOOR`` logits so they rarely win but
    the distribution still normalizes like a categorical over four labels.
    """
    letters_t = tuple(letters)
    logits = []
    for L in letters_t:
        v = float(logp_map.get(L, _LOGPROB_FLOOR))
        logits.append(v)
    logits_arr = np.array(logits, dtype=np.float64)
    probs = softmax(logits_arr)
    probs = np.clip(probs, eps_prob, 1.0)
    probs = probs / probs.sum()
    return probs


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    m = np.max(logits)
    ex = np.exp(logits - m)
    return ex / np.sum(ex)


def equation7_prior_from_rollouts(per_perm_label_probs: np.ndarray) -> np.ndarray:
    """Eq.~(7): softmax( mean_I log P_obs(di | q, x^I) )."""
    probs = np.clip(per_perm_label_probs, 1e-12, 1.0)
    log_probs = np.log(probs)
    avg_log = np.mean(log_probs, axis=0)
    return softmax(avg_log.astype(np.float64))


def equation1_cyclic_debiased_content_probs(
        per_perm_label_probs: np.ndarray,
) -> np.ndarray:
    """Eq.~(1) debiased mass over canonical content slots (aligned with A..D).

    ``per_perm_label_probs[k, j]`` = P(observed picks letter ``OPTION_LETTERS[j]``
    ``|`` cyclic permutation ``k``), matching :class:`PermutationRunner`'s
    rotations over option *text*.
    """
    n = per_perm_label_probs.shape[0]
    if per_perm_label_probs.shape[1] != n:
        raise ValueError(
            f"Expect square rollout matrix ({n}x{n}), got {per_perm_label_probs.shape}"
        )
    out = np.zeros(n, dtype=np.float64)
    for canon_idx in range(n):
        acc = 0.0
        for k in range(n):
            letter_idx = (canon_idx - k) % n
            acc += float(per_perm_label_probs[k, letter_idx])
        out[canon_idx] = acc / n
    s = float(out.sum())
    if s > 0:
        out /= s
    return out


def equation8_debiased_content_probs(
        default_label_probs: np.ndarray,
        peprior_probs: np.ndarray,
        *,
        eps: float = 1e-12,
) -> np.ndarray:
    """Eq.~(8): P_debiased(oi | q,x) proportional to P_obs(di|q,x)/P_eprior(di)."""
    num = np.array(default_label_probs, dtype=np.float64, copy=True)
    den = np.array(peprior_probs, dtype=np.float64, copy=True)
    den = np.clip(den, eps, None)
    num = np.clip(num, eps, None)
    w = num / den
    sw = float(w.sum())
    if sw <= 0:
        return softmax(np.ones_like(w))
    return w / sw


def average_prior_probability_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """Mean of per-sample Eq.~(7) priors, renormalized."""
    if not vectors:
        u = np.ones(len(OPTION_LETTERS), dtype=np.float64) / len(OPTION_LETTERS)
        return u
    stacked = np.stack(vectors, axis=0)
    m = np.mean(stacked, axis=0)
    m = np.clip(m, 1e-12, None)
    m = m / m.sum()
    return m


def dict_probs_to_ordered(prob_map: Mapping[str, float]) -> np.ndarray:
    return np.array([float(prob_map.get(L, 0.0)) for L in OPTION_LETTERS], dtype=np.float64)


def ordered_probs_to_dict(vec: np.ndarray) -> dict[str, float]:
    return {L: float(vec[i]) for i, L in enumerate(OPTION_LETTERS)}


@dataclass(frozen=True, slots=True)
class CalibrationState:
    """Global prior ``P_eprior(di)`` for Eq.~(8)."""

    peprior_probs: dict[str, float]
    epsilon: float = 1e-12
    estimation_question_ids: tuple[str, ...] = ()

    version: str = "v2-pride-iclr2024"


def calibration_state_uniform() -> CalibrationState:
    uni = float(1.0 / len(OPTION_LETTERS))
    return CalibrationState(
        peprior_probs={L: uni for L in OPTION_LETTERS},
        estimation_question_ids=(),
    )


def calibration_state_from_sidecar(blob: Mapping[str, Any]) -> CalibrationState:
    probs = blob.get("peprior_probs")
    if not isinstance(probs, Mapping):
        return calibration_state_uniform()
    pmap = {L: float(probs[L]) for L in OPTION_LETTERS if L in probs}
    for L in OPTION_LETTERS:
        pmap.setdefault(L, 1e-6)
    vec = dict_probs_to_ordered(pmap)
    vec = vec / vec.sum()
    ids = blob.get("calibration_question_ids") or blob.get("estimation_question_ids")
    if ids is None:
        ids_tuple: tuple[str, ...] = ()
    elif isinstance(ids, list):
        ids_tuple = tuple(str(x) for x in ids)
    else:
        ids_tuple = ()
    return CalibrationState(
        peprior_probs=ordered_probs_to_dict(vec),
        epsilon=float(blob.get("epsilon", 1e-12)),
        estimation_question_ids=ids_tuple,
        version=str(blob.get("version", "v2-pride-iclr2024")),
    )


def calibration_state_to_sidecar_payload(
        state: CalibrationState,
        *,
        calibration_seed: int,
        n_options: int = 4,
) -> dict[str, Any]:
    return {
        "version": state.version,
        "calibration_seed": calibration_seed,
        "n_options": n_options,
        "estimation_question_ids": sorted(state.estimation_question_ids),
        "peprior_probs": {L: float(state.peprior_probs.get(L, 0.0)) for L in OPTION_LETTERS},
        "epsilon": state.epsilon,
    }


def apply_debiased_choice_from_defaults(
        state: CalibrationState,
        default_logp_map: Mapping[str, float],
        *,
        eps_prob: float = 1e-12,
) -> str:
    """Argmax Eq.~(8) over canonical content slots (letters A–D order)."""
    default_probs = logprob_map_to_label_distribution(dict(default_logp_map), eps_prob=eps_prob)
    pep = dict_probs_to_ordered(state.peprior_probs)
    pep = np.clip(pep, state.epsilon, None)
    pep = pep / pep.sum()
    deb_content = equation8_debiased_content_probs(default_probs, pep, eps=eps_prob)
    return OPTION_LETTERS[int(np.argmax(deb_content))]
