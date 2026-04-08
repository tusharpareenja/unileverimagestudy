class RDEError(Exception):
    """Base class for all RDE-related errors."""
    pass

class InfeasibleConfigError(RDEError):
    """
    CRITICAL: The configuration is mathematically impossible.
    Trigger: T < P (Individual regression impossible), k* < 1.0.
    Action: Hard stop. User must adjust inputs.
    """
    pass

class RowConstraintsError(RDEError):
    """
    CRITICAL: Row rules (R1/R2) are self-contradictory.
    Trigger: min_actives > max_actives.
    Action: Hard stop.
    """
    pass

class BlockConstructionError(RDEError):
    """
    RETRYABLE (Local): Failed to build a valid T x P block.
    Trigger: Greedy algorithm stuck (cannot find valid row).
    Action: Retry building this specific respondent.
    """
    pass

class BlockUniquenessError(RDEError):
    """
    RETRYABLE (Local): Generated block is identical to a previous respondent.
    Trigger: Hash collision in StackBuilder.
    Action: Discard block and retry generation for this respondent.
    """
    pass

class DiagnosticsFailedError(RDEError):
    """
    RETRYABLE (Global): Design built but failed statistical checks.
    Trigger: Individual blocks are rank deficient or High Global VIF.
    Action: Discard entire global design, re-seed, and restart.
    """
    pass
