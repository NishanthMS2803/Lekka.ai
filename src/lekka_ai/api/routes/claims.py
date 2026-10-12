"""FastAPI routes for querying individual claims and grounded evidence."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from lekka_ai.api.dependencies import (
    get_optional_db_session,
    get_run_manager,
)
from lekka_ai.api.models import ClaimResponse, claim_to_response
from lekka_ai.api.services import RunManager
from lekka_ai.db.ledger import LedgerRepository

router = APIRouter(prefix="/claims", tags=["claims"])


@router.get(
    "/{claim_id}",
    response_model=ClaimResponse,
    summary="Get claim with supporting evidence",
)
async def get_claim(
    claim_id: str,
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    session: Annotated[Session | None, Depends(get_optional_db_session)] = None,
) -> ClaimResponse:
    """Retrieve an individual claim by ID along with its verifiable evidence."""
    # Query database ledger if session is available
    if session is not None:
        try:
            ledger = LedgerRepository(session)
            db_claim = ledger.get_claim(claim_id)
            if db_claim is not None:
                return claim_to_response(db_claim)
        except Exception:
            pass

    # Fallback to in-memory run manager index
    mem_claim = run_manager.get_claim(claim_id)
    if mem_claim is not None:
        return claim_to_response(mem_claim)

    raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")
