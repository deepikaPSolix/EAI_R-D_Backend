import asyncio
import json
from biomcp.trials.search import TrialQuery, TrialPhase, search_trials
from werkzeug.exceptions import BadRequest

VALID_PHASES = {p.name for p in TrialPhase}


async def run_trial_query(
    raw_conditions: str,
    raw_phase: str | None = None,
    raw_size: int | str = 10,
) -> list[dict]:
    """
    Asynchronously fetch clinical trials that match the given parameters.
    """

    # ── 1) conditions → list[str] ─────────────────────────────────────────
    conditions = [c.strip() for c in str(raw_conditions).split(",") if c.strip()]
    if not conditions:
        raise BadRequest("At least one condition is required")

    # ── 2) phase → TrialPhase | None ──────────────────────────────────────
    phase_param = None
    if raw_phase:
        token = str(raw_phase).strip().upper()
        if token not in VALID_PHASES:
            raise BadRequest(f"Phase must be one of {', '.join(VALID_PHASES)}")
        phase_param = TrialPhase[token]

    # ── 3) page size → int (1-100) ────────────────────────────────────────
    try:
        size = int(raw_size)
    except ValueError:
        raise BadRequest("size must be an integer")
    if not 1 <= size <= 100:
        raise BadRequest("size must be between 1 and 100")

    # ── 4) fire BioMCP ────────────────────────────────────────────────────
    q = TrialQuery(conditions=conditions, phase=phase_param, page_size=size)
    raw = await search_trials(q, output_json=True)
    return json.loads(raw)


def get_trials_sync(*args, **kwargs) -> list[dict]:
    """
    Synchronous wrapper around run_trial_query – convenient for Flask routes.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_trial_query(*args, **kwargs))
    finally:
        loop.close()
