import asyncio, json,re
from werkzeug.exceptions import BadRequest
from biomcp.trials.search import TrialQuery, TrialPhase, search_trials

VALID_PHASES        = {p.name for p in TrialPhase}                
VALID_STATUSES      = {"OPEN", "CLOSED", "ANY"}
VALID_STUDY_TYPES   = {"INTERVENTIONAL", "OBSERVATIONAL"}
NCT_RX            = re.compile(r"^NCT\d{8}$", re.I)

async def run_trial_query(
    raw_conditions:       str,
    raw_phase:            str | None  = None,
    raw_interventions:    str | None  = None,
    raw_status:           str | None  = None,
    raw_study_type:       str | None  = None,
    raw_nct_ids:       str | None = None,
    raw_size:             int | str   = 10,
) -> list[dict]:
    """Fetch trials asynchronously; raises BadRequest on bad input."""

    # 1) conditions  →  list[str]
    conditions = [c.strip() for c in str(raw_conditions).split(",") if c.strip()]
    if not conditions:
        raise BadRequest("At least one condition is required")

    # 2) phase  →  TrialPhase | None
    phase_param = None
    if raw_phase:
        token = str(raw_phase).strip().upper()
        if token not in VALID_PHASES:
            raise BadRequest(f"Phase must be one of {', '.join(VALID_PHASES)}")
        phase_param = TrialPhase[token]

    # 3) interventions  →  list[str] | None
    interventions = None
    if raw_interventions:
        interventions = [i.strip() for i in raw_interventions.split(",") if i.strip()]

    # 4) recruiting status
    status_param = None
    if raw_status:
        token = raw_status.upper()
        if token not in VALID_STATUSES:
            raise BadRequest(f"recruiting_status must be one of {', '.join(VALID_STATUSES)}")
        status_param = token

    # 5) study type
    study_type_param = None
    if raw_study_type:
        token = raw_study_type.upper()
        if token not in VALID_STUDY_TYPES:
            raise BadRequest(f"study_type must be one of {', '.join(VALID_STUDY_TYPES)}")
        study_type_param = token

    # 6) NCT_IDS
    nct_ids = None
    if raw_nct_ids:
        nct_ids = [n.strip().upper() for n in raw_nct_ids.split(",") if n.strip()]
        bad = [n for n in nct_ids if not NCT_RX.match(n)]
        if bad:
            raise BadRequest(f"Invalid NCT ID(s): {', '.join(bad)}")

    # 7) page size
    try:
        size = int(raw_size)
    except ValueError:
        raise BadRequest("size must be an integer")
    if not 1 <= size <= 100:
        raise BadRequest("size must be between 1 and 100")

    # 8) build & fire
    q = TrialQuery(
        conditions         = conditions,
        phase              = phase_param,
        interventions      = interventions,
        recruiting_status  = status_param,
        study_type         = study_type_param,
        nct_ids           = nct_ids,
        page_size          = size,
    )
    raw = await search_trials(q, output_json=True)
    return json.loads(raw)

# convenient sync wrapper ­– unchanged call-signature for routes.py
def get_trials_sync(*args, **kwargs) -> list[dict]:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_trial_query(*args, **kwargs))
    finally:
        loop.close()