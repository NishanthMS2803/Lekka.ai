"""Ledger repository for persisting and querying claims and evidence."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.attributes import flag_modified

from lekka_ai.db.models import ClaimModel, EvidenceModel
from lekka_ai.schemas.claims import Claim

if TYPE_CHECKING:
    from lekka_ai.schemas.claims import ClaimType


class LedgerRepository:
    """Repository managing claims and evidence ledger storage and queries."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_claims(
        self,
        claims: Sequence[Claim | ClaimModel | dict[str, Any]],
        *,
        run_id: uuid.UUID | str,
        task_id: uuid.UUID | str | None = None,
        agent_run_id: uuid.UUID | str | None = None,
    ) -> list[ClaimModel]:
        """Persist financial claims and linked evidence into the database."""
        run_uuid = (
            uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        )
        task_uuid = (
            uuid.UUID(str(task_id))
            if task_id is not None and not isinstance(task_id, uuid.UUID)
            else task_id
        )
        agent_run_uuid = (
            uuid.UUID(str(agent_run_id))
            if agent_run_id is not None and not isinstance(agent_run_id, uuid.UUID)
            else agent_run_id
        )

        persisted: list[ClaimModel] = []

        for item in claims:
            if isinstance(item, ClaimModel):
                item.run_id = run_uuid
                if task_uuid is not None and item.task_id is None:
                    item.task_id = task_uuid
                if agent_run_uuid is not None and item.agent_run_id is None:
                    item.agent_run_id = agent_run_uuid
                for ev in item.evidence:
                    if ev.run_id is None:
                        ev.run_id = run_uuid
                    if task_uuid is not None and ev.task_id is None:
                        ev.task_id = task_uuid
                    if agent_run_uuid is not None and ev.agent_run_id is None:
                        ev.agent_run_id = agent_run_uuid
                self.session.add(item)
                persisted.append(item)
                continue

            claim_schema: Claim
            if isinstance(item, dict):
                claim_schema = Claim.model_validate(item)
            elif isinstance(item, Claim):
                claim_schema = item
            else:
                msg = f"Unsupported claim object type: {type(item)}"
                raise TypeError(msg)

            claim_model = ClaimModel(
                claim_id=claim_schema.claim_id,
                run_id=run_uuid,
                task_id=task_uuid,
                agent_run_id=agent_run_uuid,
                text=claim_schema.text,
                type=str(claim_schema.type),
                confidence=float(claim_schema.confidence),
                derived_by=claim_schema.derived_by,
                produced_by=claim_schema.produced_by,
                contradicted_by=list(claim_schema.contradicted_by or []),
                depends_on=list(claim_schema.depends_on or []),
            )

            for ev_schema in claim_schema.evidence:
                tc_uuid: uuid.UUID | None = None
                if ev_schema.tool_call_id:
                    if isinstance(ev_schema.tool_call_id, uuid.UUID):
                        tc_uuid = ev_schema.tool_call_id
                    else:
                        try:
                            tc_uuid = uuid.UUID(str(ev_schema.tool_call_id).strip())
                        except ValueError:
                            tc_uuid = None

                retrieved_at_dt: datetime | None = None
                if ev_schema.retrieved_at:
                    if isinstance(ev_schema.retrieved_at, datetime):
                        retrieved_at_dt = ev_schema.retrieved_at
                    else:
                        try:
                            retrieved_at_dt = datetime.fromisoformat(
                                str(ev_schema.retrieved_at).strip()
                            )
                        except ValueError:
                            retrieved_at_dt = None

                as_of_date_val: date | None = None
                if ev_schema.as_of_date:
                    if isinstance(ev_schema.as_of_date, date):
                        as_of_date_val = ev_schema.as_of_date
                    else:
                        try:
                            as_of_date_val = date.fromisoformat(
                                str(ev_schema.as_of_date).strip()
                            )
                        except ValueError:
                            as_of_date_val = None

                evidence_model = EvidenceModel(
                    run_id=run_uuid,
                    task_id=task_uuid,
                    agent_run_id=agent_run_uuid,
                    tool_call_id=tc_uuid,
                    source_type=ev_schema.source_type,
                    document_id=ev_schema.document_id,
                    section=ev_schema.section,
                    retrieved_at=retrieved_at_dt,
                    as_of_date=as_of_date_val,
                    raw_value=ev_schema.raw_value,
                )
                claim_model.evidence.append(evidence_model)

            self.session.add(claim_model)
            persisted.append(claim_model)

        self.session.commit()
        for model in persisted:
            self.session.refresh(model)

        return persisted

    def get_claims(
        self,
        run_id: uuid.UUID | str,
        filters: dict[str, Any] | None = None,
        *,
        claim_type: str | ClaimType | None = None,
        type: str | ClaimType | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        produced_by: str | None = None,
        task_id: uuid.UUID | str | None = None,
        agent_run_id: uuid.UUID | str | None = None,
        tool_call_id: uuid.UUID | str | None = None,
        source_type: str | None = None,
        **kwargs: Any,
    ) -> list[ClaimModel]:
        """Query claims for a run with optional filtering."""
        run_uuid = (
            uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        )

        query_filters = dict(filters or {})
        query_filters.update(kwargs)
        if claim_type is not None:
            query_filters["claim_type"] = claim_type
        if type is not None:
            query_filters["type"] = type
        if min_confidence is not None:
            query_filters["min_confidence"] = min_confidence
        if max_confidence is not None:
            query_filters["max_confidence"] = max_confidence
        if produced_by is not None:
            query_filters["produced_by"] = produced_by
        if task_id is not None:
            query_filters["task_id"] = task_id
        if agent_run_id is not None:
            query_filters["agent_run_id"] = agent_run_id
        if tool_call_id is not None:
            query_filters["tool_call_id"] = tool_call_id
        if source_type is not None:
            query_filters["source_type"] = source_type

        stmt = (
            select(ClaimModel)
            .options(joinedload(ClaimModel.evidence))
            .where(ClaimModel.run_id == run_uuid)
        )

        raw_type = query_filters.get("type") or query_filters.get("claim_type")
        if raw_type is not None:
            stmt = stmt.where(ClaimModel.type == str(raw_type).lower())

        if (
            "min_confidence" in query_filters
            and query_filters["min_confidence"] is not None
        ):
            stmt = stmt.where(
                ClaimModel.confidence >= float(query_filters["min_confidence"])
            )

        if (
            "max_confidence" in query_filters
            and query_filters["max_confidence"] is not None
        ):
            stmt = stmt.where(
                ClaimModel.confidence <= float(query_filters["max_confidence"])
            )

        if "produced_by" in query_filters and query_filters["produced_by"] is not None:
            stmt = stmt.where(
                ClaimModel.produced_by == str(query_filters["produced_by"])
            )

        if "task_id" in query_filters and query_filters["task_id"] is not None:
            t_id = query_filters["task_id"]
            t_uuid = uuid.UUID(str(t_id)) if not isinstance(t_id, uuid.UUID) else t_id
            stmt = stmt.where(ClaimModel.task_id == t_uuid)

        if (
            "agent_run_id" in query_filters
            and query_filters["agent_run_id"] is not None
        ):
            ar_id = query_filters["agent_run_id"]
            ar_uuid = (
                uuid.UUID(str(ar_id)) if not isinstance(ar_id, uuid.UUID) else ar_id
            )
            stmt = stmt.where(ClaimModel.agent_run_id == ar_uuid)

        needs_evidence_join = False
        evidence_conditions = []

        if (
            "tool_call_id" in query_filters
            and query_filters["tool_call_id"] is not None
        ):
            tc_id = query_filters["tool_call_id"]
            tc_uuid = (
                uuid.UUID(str(tc_id)) if not isinstance(tc_id, uuid.UUID) else tc_id
            )
            needs_evidence_join = True
            evidence_conditions.append(EvidenceModel.tool_call_id == tc_uuid)

        if "source_type" in query_filters and query_filters["source_type"] is not None:
            needs_evidence_join = True
            evidence_conditions.append(
                EvidenceModel.source_type == str(query_filters["source_type"])
            )

        if needs_evidence_join:
            stmt = stmt.join(ClaimModel.evidence).where(*evidence_conditions)

        stmt = stmt.order_by(
            ClaimModel.created_at.asc(), ClaimModel.id.asc()
        ).distinct()
        return list(self.session.execute(stmt).unique().scalars().all())

    def get_claim(
        self,
        claim_id: str | uuid.UUID,
        run_id: uuid.UUID | str | None = None,
    ) -> ClaimModel | None:
        """Find a claim by its string claim_id or primary key UUID."""
        stmt = select(ClaimModel).options(joinedload(ClaimModel.evidence))
        if isinstance(claim_id, uuid.UUID):
            stmt = stmt.where(ClaimModel.id == claim_id)
        else:
            str_id = str(claim_id)
            try:
                parsed_uuid = uuid.UUID(str_id)
                stmt = stmt.where(
                    or_(ClaimModel.claim_id == str_id, ClaimModel.id == parsed_uuid)
                )
            except ValueError:
                stmt = stmt.where(ClaimModel.claim_id == str_id)

        if run_id is not None:
            run_uuid = (
                uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
            )
            stmt = stmt.where(ClaimModel.run_id == run_uuid)

        return self.session.execute(stmt).unique().scalars().first()

    def mark_contradiction(
        self,
        claim_id: str | uuid.UUID,
        contradicted_by_id: str | uuid.UUID,
        *,
        run_id: uuid.UUID | str | None = None,
    ) -> ClaimModel:
        """Mark a claim as contradicted by another claim or evidence identifier."""
        claim = self.get_claim(claim_id, run_id=run_id)
        if claim is None:
            msg = f"Claim '{claim_id}' not found."
            raise ValueError(msg)

        contr_str = str(contradicted_by_id)
        current = list(claim.contradicted_by or [])
        if contr_str not in current:
            current.append(contr_str)
            claim.contradicted_by = current
            flag_modified(claim, "contradicted_by")
            self.session.commit()
            self.session.refresh(claim)

        return claim
