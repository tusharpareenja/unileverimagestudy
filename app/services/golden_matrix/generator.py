"""
Main entry: generate full N×P Golden Matrix (one block per respondent).

Preflight validates feasibility upfront. When the structural VIF is well
below threshold, expensive per-row/block VIF checks are skipped.

On diagnostics failure, max_vif is adaptively relaxed and retried.
"""
import numpy as np
import pandas as pd
import logging

from .config import DesignConfig
from .design_builder import StackBuilder
from .diagnostics import run_all_diagnostics
from .exceptions import RDEError, InfeasibleConfigError, DiagnosticsFailedError, BlockConstructionError
from .preflight import preflight_check

logger = logging.getLogger("RDE_Generator")

_VIF_RELAX_STEP = 5.0
_MAX_VIF_CEILING = 30.0


def generate_golden_matrix(
    config: DesignConfig,
    progress=None,
    progress_callback=None,
    rng_seed=None,
):
    """
    Generate the RDE Golden Matrix.

    Args:
        config:   DesignConfig with all design parameters.
        progress: optional ``[int]`` list updated with respondents completed.
        progress_callback: optional ``(done, total)`` called after each respondent.
        rng_seed: optional base seed (combined with attempt index for retries).

    Returns:
        (df, report) — DataFrame with respondent, task, and silo columns;
        diagnostics dict (includes ``X`` raw array and ``preflight`` results).
    """
    logger.info("--- Golden Matrix: N=%d rows, P=%d elements ---",
                config.total_rows, config.total_elements)

    # --- Preflight ---
    pf = preflight_check(config)
    if not pf["feasible"]:
        reasons = "; ".join(pf["warnings"])
        advice = "; ".join(pf["recommendations"]) if pf["recommendations"] else "Review config."
        raise InfeasibleConfigError(f"Preflight failed: {reasons}\nRecommendations: {advice}")
    for w in pf["warnings"]:
        logger.warning("Preflight warning: %s", w)

    skip_vif = (
        np.isfinite(pf["structural_vif"])
        and pf["structural_vif"] < config.max_vif * 0.5
    )

    # Track the current max_vif (may be relaxed on retry) without
    # re-creating the entire DesignConfig each time.
    current_max_vif = config.max_vif

    for attempt in range(config.max_attempts):
        base = int(rng_seed) if rng_seed is not None else 0
        seed = base * 1_000_003 + attempt + 42
        rng = np.random.default_rng(seed)
        if progress is not None:
            progress[0] = 0
        try:
            if config.verbose:
                logger.info("Attempt %d/%d...", attempt + 1, config.max_attempts)
            if attempt > 0 and current_max_vif > config.max_vif:
                logging.getLogger("IdeaLab").info(
                    "Golden Matrix: relaxed max_vif=%.1f (adaptive).", current_max_vif)

            # Temporarily patch max_vif for this attempt
            orig_vif = config.max_vif
            object.__setattr__(config, "max_vif", current_max_vif)

            builder = StackBuilder(
                config,
                rng,
                progress=progress,
                skip_vif=skip_vif,
                progress_callback=progress_callback,
            )
            X_all = builder.build_stack()
            report = run_all_diagnostics(X_all, config)

            # Restore original
            object.__setattr__(config, "max_vif", orig_vif)

            report["preflight"] = pf
            report["X"] = X_all

            # Build DataFrame with respondent/task index + silo-labeled columns
            df = pd.DataFrame(X_all, columns=config.column_labels)
            df.insert(0, "respondent", np.repeat(
                np.arange(1, config.n_respondents + 1), config.tasks_per_respondent))
            df.insert(1, "task", np.tile(
                np.arange(1, config.tasks_per_respondent + 1), config.n_respondents))

            logger.info("SUCCESS: Golden Matrix Generated.")
            if config.verbose:
                logger.info("Global Max VIF=%.2f", report["global_max_vif"])
            return df, report

        except DiagnosticsFailedError as e:
            object.__setattr__(config, "max_vif", orig_vif)
            err_msg = str(e)
            is_vif = "VIF" in err_msg
            if is_vif and current_max_vif < _MAX_VIF_CEILING:
                current_max_vif = min(current_max_vif + _VIF_RELAX_STEP, _MAX_VIF_CEILING)
            logging.getLogger("IdeaLab").warning(
                "Golden Matrix attempt %d/%d failed: %s. %s",
                attempt + 1, config.max_attempts, err_msg[:100],
                f"Relaxing max_vif to {current_max_vif:.1f}..." if is_vif else "Retrying...",
            )
            if config.verbose:
                logger.warning("  >> Failed Attempt %d: %s", attempt + 1, err_msg)

        except BlockConstructionError as e:
            object.__setattr__(config, "max_vif", orig_vif)
            logging.getLogger("IdeaLab").warning(
                "Golden Matrix attempt %d/%d failed: %s. Retrying...",
                attempt + 1, config.max_attempts, str(e)[:120],
            )
            if config.verbose:
                logger.warning("  >> Failed Attempt %d: %s", attempt + 1, str(e))

    raise RDEError(f"Could not generate valid matrix after {config.max_attempts} attempts.")
