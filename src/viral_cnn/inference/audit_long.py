"""
Audit utilities for long-sequence inference, logit equivalence, and training overlap.
"""

from typing import Any, Optional, Sequence
from collections import Counter
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .engine import predict_sequence
from .voter import LongSequenceVoter

DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Returns the reverse complement of a DNA sequence."""
    return seq.translate(DNA_COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    """Returns lexicographically smallest between forward and reverse complement kmer."""
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def verify_window_logit_equivalence(
    model: nn.Module,
    sequence_512: str,
    device: str = "cpu",
) -> float:
    """
    Verifies that a 512 bp window produces numerically identical logits
    when processed via the single-fragment engine vs the LongSequenceVoter.
    Returns the maximum absolute difference between logit vectors.
    """
    if len(sequence_512) != 512:
        raise ValueError(f"Expected sequence of exactly 512 bp, got {len(sequence_512)}")

    # 1. Single-fragment engine inference
    single_res = predict_sequence(model, sequence_512, device=device, max_len=512)
    single_logits = torch.tensor(single_res.logits, dtype=torch.float32)

    # 2. Voter sliding-window inference with window=512, stride=512
    voter = LongSequenceVoter(
        model=model,
        window_size=512,
        stride=512,
        handle_short="pad",
        device=device,
    )
    voter_res = voter.predict_sequence(sequence_512)
    if voter_res is None:
        raise RuntimeError("Voter returned None for 512 bp sequence")

    # To get raw logits directly from voter pass:
    tokens = torch.tensor(
        voter.model.one_hot_layer.num_features if hasattr(voter.model, "one_hot_layer") else 4
    )
    # Forward pass of voter's exact window
    from .voter import slice_sequence

    windows = slice_sequence(sequence_512, window_size=512, stride=512, handle_short="pad")
    tokens_tensor = torch.tensor(windows, dtype=torch.long, device=device)
    with torch.no_grad():
        voter_logits = voter.model(tokens_tensor).cpu().squeeze(0)

    max_abs_diff = torch.max(torch.abs(single_logits - voter_logits)).item()
    return float(max_abs_diff)


def quantify_training_overlap(
    test_sequences: list[str],
    train_sequences: list[str],
    k: int = 15,
) -> dict[str, float]:
    """
    Quantifies exact sequence and canonical k-mer overlap between test sequences
    and training reference fragments.

    Returns:
    - exact_match_fraction: fraction of test sequences having identical (or RC) match in training.
    - mean_kmer_overlap_fraction: average fraction of test k-mers present in training k-mer set.
    - fraction_with_any_overlap: fraction of test sequences sharing at least 1 canonical k-mer.
    """
    if not test_sequences or not train_sequences:
        return {
            "exact_match_fraction": 0.0,
            "mean_kmer_overlap_fraction": 0.0,
            "fraction_with_any_overlap": 0.0,
        }

    # Build canonical training set
    train_exact = set()
    train_kmers = set()

    for seq in train_sequences:
        seq_clean = seq.upper().strip()
        train_exact.add(seq_clean)
        train_exact.add(reverse_complement(seq_clean))

        if len(seq_clean) >= k:
            for i in range(len(seq_clean) - k + 1):
                kmer = seq_clean[i : i + k]
                if all(c in "ACGT" for c in kmer):
                    train_kmers.add(canonical_kmer(kmer))

    exact_matches = 0
    kmer_overlaps = []
    any_overlap_count = 0

    for test_seq in test_sequences:
        test_clean = test_seq.upper().strip()

        # Check exact sequence match
        if test_clean in train_exact or reverse_complement(test_clean) in train_exact:
            exact_matches += 1

        # Check k-mer overlap
        test_seq_kmers = set()
        if len(test_clean) >= k:
            for i in range(len(test_clean) - k + 1):
                kmer = test_clean[i : i + k]
                if all(c in "ACGT" for c in kmer):
                    test_seq_kmers.add(canonical_kmer(kmer))

        if test_seq_kmers:
            shared = test_seq_kmers.intersection(train_kmers)
            overlap_frac = len(shared) / len(test_seq_kmers)
            kmer_overlaps.append(overlap_frac)
            if len(shared) > 0:
                any_overlap_count += 1
        else:
            kmer_overlaps.append(0.0)

    n_test = len(test_sequences)
    return {
        "exact_match_fraction": float(exact_matches / n_test),
        "mean_kmer_overlap_fraction": float(np.mean(kmer_overlaps) if kmer_overlaps else 0.0),
        "fraction_with_any_overlap": float(any_overlap_count / n_test),
    }


def stratify_long_errors(
    df_results: pd.DataFrame,
    length_bins: Optional[list[tuple[int, int, str]]] = None,
    ambiguity_bins: Optional[list[tuple[float, float, str]]] = None,
    taxon_col: str = "taxon_id",
) -> dict[str, Any]:
    """
    Stratifies prediction performance across sequence length bins and ambiguity fractions.

    Expects df_results to contain columns:
    - 'length': integer sequence length
    - 'ambiguity_ratio': fraction of non-ACGT characters
    - 'truth_label': integer true class (1-based or 0-based)
    - 'pred_mean_logits': integer predicted class
    - 'pred_majority_vote': integer predicted class
    """
    if length_bins is None:
        length_bins = [
            (0, 512, "len_lt_512"),
            (512, 2000, "len_512_to_2000"),
            (2000, 5000, "len_2000_to_5000"),
            (5000, 10001, "len_5000_to_10000"),
            (10001, float("inf"), "len_gt_10000"),
        ]

    if ambiguity_bins is None:
        ambiguity_bins = [
            (0.0, 0.0, "ambig_0_pct"),
            (0.0, 0.05, "ambig_0_to_5_pct"),
            (0.05, 1.0, "ambig_gt_5_pct"),
        ]

    if length_bins:
        ordered_length_bins = sorted(length_bins, key=lambda item: item[0])
        if ordered_length_bins[0][0] != 0 or ordered_length_bins[-1][1] != float("inf"):
            raise ValueError("length strata must cover [0, infinity)")
        if any(left[1] != right[0] for left, right in zip(ordered_length_bins, ordered_length_bins[1:])):
            raise ValueError("length strata must be disjoint and contiguous")
    if ambiguity_bins:
        zero_bins = [b for b in ambiguity_bins if b[0] == b[1] == 0.0]
        intervals = sorted([b for b in ambiguity_bins if b[0] != b[1]], key=lambda item: item[0])
        if len(zero_bins) != 1 or not intervals or intervals[0][0] != 0.0 or intervals[-1][1] != 1.0:
            raise ValueError("ambiguity strata must contain exact zero and cover (0, 1]")
        if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("ambiguity strata must be disjoint and contiguous")
    if "length" in df_results:
        lengths = pd.to_numeric(df_results["length"], errors="coerce")
        if lengths.isna().any() or (lengths < 0).any():
            raise ValueError("length values must be known non-negative numbers")
    if "ambiguity_ratio" in df_results:
        ratios = pd.to_numeric(df_results["ambiguity_ratio"], errors="coerce")
        invalid = ratios.notna() & ((ratios < 0) | (ratios > 1))
        if invalid.any():
            raise ValueError("ambiguity_ratio values must be within [0, 1] or missing")

    # Calculate correctness flags
    mean_correct = df_results["truth_label"] == df_results["pred_mean_logits"]
    majority_correct = df_results["truth_label"] == df_results["pred_majority_vote"]

    strata: dict[str, Any] = {
        "overall": {
            "total_samples": int(len(df_results)),
            "correct_mean_logits": int(mean_correct.sum()),
            "correct_majority_vote": int(majority_correct.sum()),
            "accuracy_mean_logits": float(mean_correct.mean()) if len(df_results) else None,
            "accuracy_majority_vote": float(majority_correct.mean()) if len(df_results) else None,
        },
        "by_length": {},
        "by_ambiguity": {},
        "by_taxon": {},
    }

    # Stratify by length
    for low, high, label in length_bins:
        mask = (df_results["length"] >= low) & (df_results["length"] < high)
        subset = df_results[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else None
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else None
        strata["by_length"][label] = {
            "range": [low, high],
            "count": count,
            "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
            "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }

    # Stratify by ambiguity
    for low, high, label in ambiguity_bins:
        if low == high == 0.0:
            mask = df_results["ambiguity_ratio"] == 0.0
        else:
            mask = (df_results["ambiguity_ratio"] > low) & (df_results["ambiguity_ratio"] <= high) if low == 0 else (df_results["ambiguity_ratio"] > low) & (df_results["ambiguity_ratio"] <= high)
        subset = df_results[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else None
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else None
        strata["by_ambiguity"][label] = {
            "range": [low, high],
            "count": count,
            "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
            "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }
    if "ambiguity_ratio" in df_results:
        subset = df_results[df_results["ambiguity_ratio"].isna()]
        count = int(len(subset))
        strata["by_ambiguity"]["ambig_unknown"] = {
            "range": None,
            "count": count,
            "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
            "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
            "accuracy_mean_logits": float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count else None,
            "accuracy_majority_vote": float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count else None,
        }

    if taxon_col in df_results:
        group_values = df_results[taxon_col].astype("object").where(df_results[taxon_col].notna(), "unknown")
        for taxon, indices in group_values.groupby(group_values, sort=True).groups.items():
            subset = df_results.loc[indices]
            count = int(len(subset))
            strata["by_taxon"][str(taxon)] = {
                "count": count,
                "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
                "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
                "accuracy_mean_logits": float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count else None,
                "accuracy_majority_vote": float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count else None,
            }

    return strata


def audit_long_inputs(
    queries: pd.DataFrame,
    *,
    sequence_col: str = "sequence",
    query_id_col: str = "query_id",
    truth_col: str = "truth_label",
    taxon_col: str = "truth_label",
    genome_id_col: Optional[str] = None,
    source_path: Optional[str] = None,
    data_kind: Optional[str] = None,
    excluded_count: int = 0,
    exclusion_rule: str = "caller-declared; no implicit exclusions",
    exclusion_counts: Optional[dict[str, int]] = None,
    sample_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Describe the rows actually observed; absent genome IDs remain unknown."""
    if sequence_col not in queries:
        raise ValueError(f"Missing sequence column: {sequence_col}")
    ids = queries[query_id_col].astype(str).tolist() if query_id_col in queries else [str(i) for i in queries.index]
    if len(ids) != len(set(ids)):
        raise ValueError("query IDs must be unique")
    sequences = queries[sequence_col].fillna("").astype(str)
    lengths = sequences.str.len()
    ambiguity = sequences.map(lambda s: sum(c.upper() not in "ACGT" for c in s) / len(s) if s else None)
    groups = queries[taxon_col].astype(str).value_counts().to_dict() if taxon_col in queries else {}
    genomes_known = bool(genome_id_col and genome_id_col in queries and queries[genome_id_col].notna().all())
    source_hash = None
    source_identities = []
    if source_path:
        h = hashlib.sha256()
        source = __import__("pathlib").Path(source_path)
        source_files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        for file_path in source_files:
            relative = str(file_path.relative_to(source) if source.is_dir() else file_path.name)
            file_hash = hashlib.sha256()
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    file_hash.update(chunk)
            digest = file_hash.hexdigest()
            h.update(relative.encode()); h.update(digest.encode())
            source_identities.append({"path": str(file_path.resolve()), "sha256": digest, "size_bytes": file_path.stat().st_size})
        source_hash = h.hexdigest()
    return {
        "source_path": source_path,
        "source_sha256": source_hash,
        "source_files": source_identities,
        "data_kind": data_kind or ("contigs" if source_path and "ncbi" in source_path.lower() else "unknown"),
        "available_count": int((sample_metadata or {}).get("available_count", (sample_metadata or {}).get("population_count", len(queries)) + excluded_count)),
        "eligible_count": int((sample_metadata or {}).get("population_count", len(queries))),
        "observed_count": int(len(queries)),
        "excluded_count": int(excluded_count),
        "exclusion_rule": exclusion_rule,
        "exclusion_counts": exclusion_counts or {},
        "length": {"min": int(lengths.min()) if len(lengths) else None, "max": int(lengths.max()) if len(lengths) else None, "mean": float(lengths.mean()) if len(lengths) else None},
        "ambiguity": {"mean": float(ambiguity.mean()) if ambiguity.notna().any() else None, "missing_count": int(ambiguity.isna().sum())},
        "taxon_counts": {str(k): int(v) for k, v in groups.items()},
        "genome_identity": {"status": "known" if genomes_known else "unknown", "genome_count": int(queries[genome_id_col].nunique()) if genomes_known else None},
        "query_ids": ids,
        "sampling": sample_metadata or {"kind": "census", "planned_count": int(len(queries)), "realized_count": int(len(queries)), "seed": None, "selected_query_ids": ids},
        "historical_sampling_rule": {"small_lt_10000": 50, "medium_10000_to_99999": 200, "large_ge_100000": 500, "status": "historical rule; observed counts measured separately"},
    }


def sample_long_queries(queries: pd.DataFrame, limit: Optional[int] = None, seed: int = 42, query_id_col: str = "query_id") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministic bounded sample, retaining the selected query identities."""
    if limit is None or limit >= len(queries):
        selected = queries.copy()
        rule = "census"
    else:
        selected = queries.sample(n=max(0, int(limit)), random_state=seed).sort_index()
        rule = "pandas sample without replacement, sorted by original index"
    ids = selected[query_id_col].astype(str).tolist() if query_id_col in selected else [str(i) for i in selected.index]
    return selected, {"kind": "sample" if len(selected) < len(queries) else "census", "population_count": int(len(queries)), "planned_count": limit, "realized_count": len(selected), "criterion": rule, "seed": seed if rule != "census" else None, "selected_query_ids": ids}


def quantify_training_overlap_detailed(test_sequences: Sequence[str], train_sequences: Sequence[str], k: int = 15) -> dict[str, Any]:
    """C02 overlap contract with independent universes and explicit empty-kmer support."""
    base = quantify_training_overlap(list(test_sequences), list(train_sequences), k=k)
    valid_queries = 0
    query_kmers = 0
    shared_kmers = 0
    train_kmer_set = set()
    for raw in train_sequences:
        seq = raw.upper().strip()
        for i in range(max(0, len(seq)-k+1)):
            kmer=seq[i:i+k]
            if len(kmer)==k and all(c in "ACGT" for c in kmer): train_kmer_set.add(canonical_kmer(kmer))
    for raw in test_sequences:
        seq=raw.upper().strip(); kmers={canonical_kmer(seq[i:i+k]) for i in range(max(0,len(seq)-k+1)) if len(seq[i:i+k])==k and all(c in "ACGT" for c in seq[i:i+k])}
        if kmers:
            valid_queries += 1; query_kmers += len(kmers); shared_kmers += len(kmers & train_kmer_set)
    return {**base, "k": int(k), "canonicalization": "canonicalized: lexicographic min(forward, reverse complement)", "train_universe_count": len(train_sequences), "query_universe_count": len(test_sequences), "train_valid_unique_kmers": len(train_kmer_set), "query_valid_kmer_denominator": query_kmers, "query_shared_unique_kmers_numerator": shared_kmers, "queries_with_valid_kmers": int(valid_queries), "queries_without_valid_kmers": int(len(test_sequences)-valid_queries), "query_denominator": len(test_sequences), "train_denominator": len(train_sequences), "sampling": {"train": "caller-declared", "queries": "caller-declared"}}


def shared_noise_conditions(contigs: Sequence[tuple], noise_rates: Sequence[float], seed: int = 42) -> dict[float, list[tuple]]:
    """Corrupt each full query once so every model sees the same realization before windowing."""
    from viral_cnn.eval.noise import inject_noise
    from .voter import DNA_MAP
    inverse = {v:k for k,v in DNA_MAP.items()}
    output: dict[float, list[tuple]] = {0.0: list(contigs)}
    for rate in sorted(set(float(x) for x in noise_rates)):
        if rate <= 0 or rate > 1:
            raise ValueError("optional C02 noise rates must be in (0, 1]")
        rows = []
        for record in contigs:
            qid, sequence, *metadata = record
            digest = hashlib.sha256(f"{seed}:{rate:.12g}:{qid}".encode()).digest()
            local_seed = int.from_bytes(digest[:8], "big") % (2**63-1)
            tokens = torch.tensor([[DNA_MAP.get(base.upper(), 4) for base in sequence]], dtype=torch.long)
            generator = torch.Generator(device="cpu").manual_seed(local_seed)
            noisy = inject_noise(tokens, rate, generator=generator)[0].tolist()
            corrupted = "".join(inverse.get(int(token), "N") for token in noisy)
            rows.append((qid, corrupted, *metadata))
        output[rate] = rows
    return output


def build_diagnostic_finding(claim: str, test: str, effect: Any, n: int, inputs: Sequence[str], limitations: Sequence[str], *, status: str, is_smoke_test: bool = False) -> dict[str, Any]:
    allowed = {"SUPPORTED", "HYPOTHESIS", "INCONCLUSIVE"}
    if status not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")
    if is_smoke_test and status == "SUPPORTED":
        raise ValueError("synthetic smoke data cannot support an empirical finding")
    return {"status": status, "claim": claim, "test": test, "effect": effect, "n": int(n), "inputs": list(inputs), "limitations": list(limitations)}


def map_labels_to_canonical(labels: Sequence[Any], mapping: pd.DataFrame, *, source_id_col: str, canonical_id_col: str, canonical_order: Sequence[Any]) -> tuple[list[int], list[Any]]:
    """Resolve local labels using explicit IDs and a declared canonical class order."""
    if not source_id_col or not canonical_id_col or source_id_col not in mapping or canonical_id_col not in mapping:
        raise ValueError("identified source and canonical mapping columns are required")
    if mapping[source_id_col].duplicated().any():
        raise ValueError("ambiguous source label mapping")
    order = {value: i for i, value in enumerate(canonical_order)}
    if len(order) != len(canonical_order):
        raise ValueError("canonical class order contains duplicates")
    map_by_source = dict(zip(mapping[source_id_col], mapping[canonical_id_col]))
    unknown = [value for value in labels if value not in map_by_source or map_by_source[value] not in order]
    if unknown:
        raise ValueError(f"unmapped or out-of-domain source labels: {unknown}")
    return [order[map_by_source[value]] for value in labels], list(labels)


def diagnose_window_logits(window_logits: Optional[Sequence[Sequence[float]]], truth_label: Optional[int] = None, *, is_smoke_test: bool = False) -> dict[str, Any]:
    """Measure aggregation disagreement, logit scale/margins and leave-one-window-out flips."""
    if window_logits is None:
        return {"status": "INCONCLUSIVE", "reason": "raw window logits unavailable", "n_windows": 0}
    x = torch.as_tensor(window_logits, dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {"status": "INCONCLUSIVE", "reason": "no usable window logits", "n_windows": 0}
    mean_pred = int(x.mean(0).argmax())
    winners = x.argmax(1)
    vote_pred = int(torch.mode(winners).values)
    sorted_logits = x.sort(dim=1, descending=True).values
    margins = sorted_logits[:, 0] - sorted_logits[:, 1] if x.shape[1] > 1 else sorted_logits[:, 0]
    loo = []
    if len(x) > 1:
        loo = [int(torch.cat((x[:i], x[i+1:]), 0).mean(0).argmax()) for i in range(len(x))]
    return {"status": "INCONCLUSIVE" if is_smoke_test else "SUPPORTED", "n_windows": int(len(x)), "mean_prediction": mean_pred, "vote_prediction": vote_pred, "aggregation_disagrees": mean_pred != vote_pred, "mean_abs_logit": float(x.abs().mean()), "max_abs_logit": float(x.abs().max()), "mean_margin": float(margins.mean()), "leave_one_window_out_predictions": loo, "leave_one_window_out_flips": int(sum(p != mean_pred for p in loo)), "truth_label": truth_label, "correct": None if truth_label is None else mean_pred == truth_label, "limitations": ["synthetic smoke data cannot support empirical findings"] if is_smoke_test else []}


def compare_query_and_taxon_accuracy(results: pd.DataFrame, *, truth_col: str = "truth_label", prediction_col: str = "pred_mean_logits", taxon_col: str = "taxon_id", query_id_col: str = "query_id", is_smoke_test: bool = False) -> dict[str, Any]:
    required = {truth_col, prediction_col, taxon_col, query_id_col}
    if missing := sorted(required - set(results.columns)):
        raise ValueError(f"composition audit missing columns: {missing}")
    if results[query_id_col].astype(str).duplicated().any():
        raise ValueError("composition audit requires unique query IDs")
    valid = results[truth_col].notna() & results[prediction_col].notna()
    observed = results.loc[valid].copy()
    observed["correct"] = observed[truth_col].astype(int).eq(observed[prediction_col].astype(int))
    by_taxon = observed.groupby(taxon_col, dropna=False)["correct"].agg(["sum", "count", "mean"]).reset_index()
    query_acc = float(observed["correct"].mean()) if len(observed) else None
    taxon_macro = float(by_taxon["mean"].mean()) if len(by_taxon) else None
    return {"status": "INCONCLUSIVE" if is_smoke_test or not len(observed) else "SUPPORTED", "n": int(len(observed)), "excluded": int(len(results)-len(observed)), "query_weighted_accuracy": query_acc, "taxon_macro_accuracy": taxon_macro, "difference_pp_taxon_minus_query": None if query_acc is None or taxon_macro is None else float((taxon_macro-query_acc)*100), "taxa": by_taxon.rename(columns={"sum":"correct","count":"n","mean":"accuracy"}).to_dict(orient="records"), "genome_attribution": "not assessed; genome IDs are not inferred", "limitations": ["synthetic smoke data cannot support empirical findings"] if is_smoke_test else []}


def calibrate_mean_logits(window_logits: Optional[Sequence[Sequence[Sequence[float]]]], truth_labels: Sequence[Optional[int]], bins: int = 15, *, is_smoke_test: bool = False) -> dict[str, Any]:
    """ECE and NLL from softmax(mean window logits), with fixed equal-width bins."""
    if bins != 15:
        raise ValueError("C02 consensus calibration uses exactly 15 bins")
    if window_logits is None or len(window_logits) != len(truth_labels):
        return {"status": "INCONCLUSIVE", "reason": "consensus logits or aligned labels unavailable", "n": 0, "bins": bins}
    logits = []
    labels = []
    for item, label in zip(window_logits, truth_labels):
        if item is None or label is None:
            continue
        arr = torch.as_tensor(item, dtype=torch.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        logits.append(arr.mean(0))
        labels.append(int(label))
    if not logits:
        return {"status": "INCONCLUSIVE", "reason": "no usable consensus logits", "n": 0, "bins": bins}
    probs = torch.softmax(torch.stack(logits), dim=1)
    y = torch.tensor(labels, dtype=torch.long)
    if torch.any((y < 0) | (y >= probs.shape[1])):
        raise ValueError("truth label outside consensus logit class domain")
    conf, pred = probs.max(1)
    correct = pred.eq(y)
    ece = 0.0
    bin_rows = []
    edges = torch.linspace(0, 1, bins + 1, dtype=conf.dtype)
    for i in range(bins):
        mask = (conf >= edges[i]) & ((conf < edges[i+1]) if i < bins-1 else (conf <= edges[i+1]))
        count = int(mask.sum())
        accuracy = float(correct[mask].double().mean()) if count else None
        mean_conf = float(conf[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy-mean_conf)
        bin_rows.append({"bin": i, "lower": float(edges[i]), "upper": float(edges[i+1]), "n": count, "accuracy": accuracy, "mean_confidence": mean_conf})
    nll = float(-torch.log(probs[torch.arange(len(y)), y].clamp_min(torch.finfo(probs.dtype).tiny)).mean())
    return {"status": "INCONCLUSIVE" if is_smoke_test else "SUPPORTED", "n": len(y), "bins": bins, "ece": float(ece), "nll": nll, "mean_confidence": float(conf.mean()), "bin_metrics": bin_rows, "probability_source": "softmax of mean pre-softmax window logits", "limitations": ["synthetic smoke data cannot support empirical findings"] if is_smoke_test else []}

