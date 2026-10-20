"""Tests for the transaction update MCP proposal tool."""

import uuid
from datetime import date
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import mcp_server.tools  # noqa: F401
from app.models.transaction import Transaction
from mcp_server.auth import CallContext
from mcp_server.registry import REGISTRY


@pytest_asyncio.fixture
async def external_ctx(test_user) -> CallContext:
    return CallContext(
        user_id=test_user.id,
        conversation_id=uuid.uuid4(),
        external=True,
    )


@pytest_asyncio.fixture
async def clp_transaction(
    session: AsyncSession,
    test_user,
    test_workspace,
    test_account,
) -> dict:
    """The 2023 CLP row that was counted as 671,500 BRL in the report."""
    transaction = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        account_id=test_account.id,
        description="Transfer to Danny Tobar",
        amount=Decimal("671500"),
        currency="CLP",
        date=date(2023, 8, 12),
        effective_date=date(2023, 8, 12),
        type="debit",
        source="manual",
        status="posted",
        amount_primary=None,
        fx_rate_used=None,
    )
    session.add(transaction)
    await session.commit()
    await session.refresh(transaction)
    return {"id": transaction.id}


async def _fresh(session: AsyncSession, transaction_id) -> Transaction:
    return (
        await session.execute(
            select(Transaction).where(Transaction.id == transaction_id)
        )
    ).scalar_one()


def test_transaction_update_tool_is_registered():
    assert "propose_update_transaction" in REGISTRY
    assert REGISTRY["propose_update_transaction"].is_proposal


async def test_preview_fx_does_not_write(
    session: AsyncSession,
    external_ctx: CallContext,
    clp_transaction: dict,
):
    handler = REGISTRY["propose_update_transaction"].handler
    result = await handler(
        session=session,
        ctx=external_ctx,
        transaction_id=str(clp_transaction["id"]),
        amount_primary=3730.56,
    )

    assert result["kind"] == "update_transaction"
    assert result["target"]["currency"] == "CLP"
    assert result["target"]["amount"] == 671500.0
    assert result["target"]["amount_primary"] is None
    assert result["changes"]["amount_primary"] == 3730.56
    assert result["derived"]["fx_rate_used"] == pytest.approx(
        3730.56 / 671500, abs=1e-10
    )

    session.expire_all()
    fresh = await _fresh(session, clp_transaction["id"])
    assert fresh.amount_primary is None
    assert fresh.fx_rate_used is None


async def test_apply_fx_writes_amount_primary_and_derived_rate(
    session: AsyncSession,
    external_ctx: CallContext,
    clp_transaction: dict,
):
    handler = REGISTRY["propose_update_transaction"].handler
    result = await handler(
        session=session,
        ctx=external_ctx,
        transaction_id=str(clp_transaction["id"]),
        amount_primary=3730.56,
        apply=True,
    )

    assert result["applied"] is True
    assert result["transaction"]["amount_primary"] == 3730.56

    session.expire_all()
    fresh = await _fresh(session, clp_transaction["id"])
    assert float(fresh.amount_primary) == 3730.56
    assert float(fresh.fx_rate_used) == pytest.approx(
        3730.56 / 671500, abs=1e-10
    )


async def test_preview_derives_primary_amount_from_rate(
    session: AsyncSession,
    external_ctx: CallContext,
    clp_transaction: dict,
):
    handler = REGISTRY["propose_update_transaction"].handler
    result = await handler(
        session=session,
        ctx=external_ctx,
        transaction_id=str(clp_transaction["id"]),
        fx_rate_used=0.0055555558,
    )

    assert result["derived"]["amount_primary"] == pytest.approx(3730.56, abs=0.01)
    session.expire_all()
    fresh = await _fresh(session, clp_transaction["id"])
    assert fresh.amount_primary is None


async def test_update_without_changes_is_refused(
    session: AsyncSession,
    external_ctx: CallContext,
    clp_transaction: dict,
):
    handler = REGISTRY["propose_update_transaction"].handler
    result = await handler(
        session=session,
        ctx=external_ctx,
        transaction_id=str(clp_transaction["id"]),
    )

    assert result["error"] == "no changes provided"


async def test_clear_fx_conflicts_with_explicit_values(
    session: AsyncSession,
    external_ctx: CallContext,
    clp_transaction: dict,
):
    handler = REGISTRY["propose_update_transaction"].handler
    result = await handler(
        session=session,
        ctx=external_ctx,
        transaction_id=str(clp_transaction["id"]),
        amount_primary=3730.56,
        clear_fx=True,
    )

    assert "clear_fx" in result["error"]
