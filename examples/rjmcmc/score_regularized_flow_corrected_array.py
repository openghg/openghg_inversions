#!/usr/bin/env python3
"""Map one frozen E2 SLURM array index to one exploration attempt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TypeAlias

from examples.rjmcmc import score_regularized_flow_corrected_exploration as experiment

MatrixRow: TypeAlias = tuple[str, str, int, str, int]

MATRICES = {
    "compile_canary": tuple(
        (
            "overfit",
            case_id,
            256,
            config_id,
            0,
        )
        for case_id in (
            "near_gaussian__two_cell__root",
            "skewed__four_cell__root",
        )
        for config_id in (
            "fisher_partial_joint",
            "fisher_observation_joint",
        )
    ),
    "overfit": tuple(
        ("overfit", case_id, 256, config_id, init_index)
        for case_id in (
            "near_gaussian__two_cell__root",
            "skewed__four_cell__root",
        )
        for config_id in ("nll_only", "fisher_partial_joint")
        for init_index in range(4)
    ),
    "standard_s4096": tuple(
        ("standard", case_id, 4_096, config_id, init_index)
        for case_id in experiment.SELECTED_CASES
        for config_id in (
            "nll_only",
            "fisher_partial_joint",
            "nll_pretrain_then_partial",
        )
        for init_index in range(4)
    ),
    "observation_canary": tuple(
        ("standard", case_id, 4_096, "fisher_observation_joint", init_index)
        for case_id in (
            "near_gaussian__two_cell__root",
            "skewed__four_cell__root",
        )
        for init_index in range(4)
    ),
    "standard_s16384_nll": tuple(
        ("standard", case_id, 16_384, "nll_only", init_index)
        for case_id in experiment.SELECTED_CASES
        for init_index in range(4)
    ),
    "standard_s16384_partial": tuple(
        (
            "standard",
            case_id,
            16_384,
            "fisher_partial_joint",
            init_index,
        )
        for case_id in experiment.SELECTED_CASES
        for init_index in range(4)
    ),
    "standard_s16384_pretrain": tuple(
        (
            "standard",
            case_id,
            16_384,
            "nll_pretrain_then_partial",
            init_index,
        )
        for case_id in experiment.SELECTED_CASES
        for init_index in range(4)
    ),
}

EXPECTED_MATRIX_ATTEMPT_COUNTS = {
    "compile_canary": 4,
    "overfit": 16,
    "standard_s4096": 36,
    "observation_canary": 8,
    "standard_s16384_nll": 12,
    "standard_s16384_partial": 12,
    "standard_s16384_pretrain": 12,
}


def frozen_matrix_identity(
    matrix_id: str,
    array_task_id: int,
) -> dict[str, object]:
    """Return the exact committed array mapping for one task."""
    if matrix_id not in MATRICES:
        raise ValueError("matrix_id is not in the frozen catalogue.")
    matrix = MATRICES[matrix_id]
    if not 0 <= array_task_id < len(matrix):
        raise ValueError("array_task_id is outside the frozen matrix.")
    return {
        "matrix_id": matrix_id,
        "array_task_id": array_task_id,
        "array_task_count": len(matrix),
        "row": list(matrix[array_task_id]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-id", choices=tuple(MATRICES), required=True)
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--base-seed", type=int, default=731)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--oracle-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    arguments = parser.parse_args()
    raw_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_task_id is None:
        raise RuntimeError("corrected array launcher requires SLURM_ARRAY_TASK_ID.")
    task_id = int(raw_task_id)
    matrix = MATRICES[arguments.matrix_id]
    if not 0 <= task_id < len(matrix):
        raise ValueError("SLURM_ARRAY_TASK_ID is outside the frozen matrix.")
    expected_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if expected_count is not None and int(expected_count) != len(matrix):
        raise ValueError("SLURM array task count differs from the frozen matrix.")
    mode, case_id, sample_count, config_id, init_index = matrix[task_id]
    matrix_identity = frozen_matrix_identity(arguments.matrix_id, task_id)
    attempt_arguments = argparse.Namespace(
        mode=mode,
        case_id=case_id,
        sample_count=sample_count,
        config_id=config_id,
        init_index=init_index,
        attempt_tag=arguments.attempt_tag,
        base_seed=arguments.base_seed,
        source_git_revision=arguments.source_git_revision,
        oracle_bundle=arguments.oracle_bundle,
        output_root=arguments.output_root,
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        microbatch_size=arguments.microbatch_size,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        matrix_id=arguments.matrix_id,
        array_task_id=task_id,
        array_task_count=len(matrix),
        matrix_task_id=task_id,
        matrix_task_count=len(matrix),
        matrix_row=matrix[task_id],
        matrix_identity=matrix_identity,
    )
    slug = f"{mode}__{case_id}__S{sample_count}__{config_id}__init{init_index}__{arguments.attempt_tag}"
    attempt_root = arguments.output_root / "attempts" / slug
    if attempt_root.exists():
        raise FileExistsError("array task refuses to replace an existing attempt.")
    attempt_root.mkdir(parents=True)
    try:
        experiment.run_attempt(attempt_arguments, attempt_root)
    except Exception as error:
        experiment._atomic_json(
            attempt_root / "FAILURE.json",
            {
                "schema": experiment.SCHEMA,
                "matrix_id": arguments.matrix_id,
                "array_task_id": task_id,
                "array_task_count": len(matrix),
                "matrix_identity": matrix_identity,
                "source_git_revision": arguments.source_git_revision,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "completion_marker_published": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
