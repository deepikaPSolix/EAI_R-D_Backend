import asyncio
import json
from biomcp.trials.search import TrialQuery, search_trials

async def run_trial_query(condition: str, phase: str | None = None, size: int = 10) -> list[dict]:
    """
    Asynchronously fetches clinical trials matching the given parameters.
    """
    q = TrialQuery(condition=condition, phase=phase, page_size=size)
    raw = await search_trials(q, output_json=True)
    return json.loads(raw)

def get_trials_sync(condition: str, phase: str | None = None, size: int = 10) -> list[dict]:
    """
    Synchronous wrapper around run_trial_query, for use in Flask routes.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_trial_query(condition, phase, size))
    finally:
        loop.close()
