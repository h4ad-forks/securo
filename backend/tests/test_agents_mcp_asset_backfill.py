"""Tests for the asset purchase backfill proposal tool."""

import uuid
from datetime import date
from decimal import Decimal

import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import mcp_server.tools  # noqa: F401
from app.models.asset import Asset
from app.models.asset_value import AssetValue
from app.models.transaction import Transaction
from mcp_server.auth import CallContext
from mcp_server.registry import REGISTRY


@pytest_asyncio.fixture
async def ctx(test_user) -> CallContext:
    return CallContext(
        user_id=test_user.id,
        conversation_id=uuid.uuid4(),
        external=True,
    )


@pytest_asyncio.fixture
async def manual_asset(session: AsyncSession, test_user, test_workspace) -> Asset:
    asset = Asset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        name="AAPL34",
        type="investment",
        currency="BRL",
        valuation_method="manual",
        ticker="AAPL34",
        source="manual",
    )
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    return asset


@pytest_asyncio.fixture
async def bdr_transactions(
    session: AsyncSession, test_user, test_workspace, test_account
) -> list[Transaction]:
    rows = [
        Transaction(
            id=uuid.uuid4(),
            user_id=test_user.id,
            workspace_id=test_workspace.id,
            account_id=test_account.id,
            description="Compra de BDR - AAPL34",
            amount=Decimal("100.00"),
            currency="BRL",
            date=date(2025, 1, 22),
            type="debit",
            source="ofx",
            status="posted",
        ),
        Transaction(
            id=uuid.uuid4(),
            user_id=test_user.id,
            workspace_id=test_workspace.id,
            account_id=test_account.id,
            description="Compra de BDR - AAPL34",
            amount=Decimal("75.00"),
            currency="BRL",
            date=date(2025, 2, 10),
            type="debit",
            source="ofx",
            status="posted",
        ),
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def _value_count(session: AsyncSession, asset_id: uuid.UUID) -> int:
    return (
        await session.scalar(
            select(func.count()).select_from(AssetValue).where(AssetValue.asset_id == asset_id)
        )
    ) or 0


async def test_preview_explicit_lots_does_not_write(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    result = await handler(
        session=session,
        ctx=ctx,
        lots=[
            {"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00},
            {"ticker": "AAPL34", "date": "2025-02-10", "amount": 50.00},
        ],
        apply=False,
    )

    assert result["applied"] is False
    assert result["errors"] == []
    item = result["items"][0]
    assert item["proposed_purchase_date"] == "2025-01-22"
    assert item["proposed_purchase_price"] == 150.0
    assert [point["amount"] for point in item["value_points"]] == [100.0, 150.0]
    assert all(point["action"] == "insert" for point in item["value_points"])

    await session.refresh(manual_asset)
    assert manual_asset.purchase_date is None
    assert manual_asset.purchase_price is None
    assert await _value_count(session, manual_asset.id) == 0


async def test_apply_writes_purchase_fields_and_values(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    result = await handler(
        session=session,
        ctx=ctx,
        lots=[
            {"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00},
            {"ticker": "AAPL34", "date": "2025-02-10", "amount": 50.00},
        ],
        apply=True,
    )

    assert result["applied"] is True
    assert result["updated_assets"] == 1
    assert result["inserted_values"] == 2
    assert result["skipped_values"] == 0

    await session.refresh(manual_asset)
    assert manual_asset.purchase_date == date(2025, 1, 22)
    assert manual_asset.purchase_price == Decimal("150.00")
    assert manual_asset.purchase_price_primary == Decimal("150.00")

    values = (
        (
            await session.execute(
                select(AssetValue)
                .where(AssetValue.asset_id == manual_asset.id)
                .order_by(AssetValue.date)
            )
        )
        .scalars()
        .all()
    )
    assert [float(value.amount) for value in values] == [100.0, 150.0]
    assert all(value.source == "manual" for value in values)
    assert all(value.workspace_id is not None for value in values)


async def test_apply_is_idempotent(session: AsyncSession, ctx: CallContext, manual_asset: Asset):
    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    lots = [
        {"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00},
        {"ticker": "AAPL34", "date": "2025-02-10", "amount": 50.00},
    ]

    first = await handler(session=session, ctx=ctx, lots=lots, apply=True)
    second = await handler(session=session, ctx=ctx, lots=lots, apply=True)

    assert first["inserted_values"] == 2
    assert second["inserted_values"] == 0
    assert second["skipped_values"] == 2
    assert await _value_count(session, manual_asset.id) == 2


async def test_refuses_ledger_backed_asset(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    manual_asset.average_price = Decimal("10.00")
    await session.commit()

    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    result = await handler(
        session=session,
        ctx=ctx,
        lots=[{"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00}],
        apply=True,
    )

    assert result["updated_assets"] == 0
    assert result["inserted_values"] == 0
    assert any("ledger-backed" in error for error in result["errors"])
    await session.refresh(manual_asset)
    assert manual_asset.purchase_date is None


async def test_provider_synced_asset_requires_force(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    manual_asset.source = "pluggy"
    await session.commit()

    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    blocked = await handler(
        session=session,
        ctx=ctx,
        lots=[{"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00}],
        apply=True,
    )
    assert blocked["updated_assets"] == 0
    assert any("force=true" in error for error in blocked["errors"])

    forced = await handler(
        session=session,
        ctx=ctx,
        lots=[{"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00}],
        apply=True,
        force=True,
    )
    assert forced["updated_assets"] == 1
    assert forced["inserted_values"] == 1


async def test_derives_lots_from_matching_transactions(
    session: AsyncSession,
    ctx: CallContext,
    manual_asset: Asset,
    bdr_transactions: list[Transaction],
):
    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    result = await handler(session=session, ctx=ctx, apply=False)

    assert result["errors"] == []
    item = result["items"][0]
    assert item["lot_count"] == 2
    assert [point["amount"] for point in item["value_points"]] == [100.0, 175.0]


async def test_existing_purchase_fields_conflict_without_overwrite(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    manual_asset.purchase_date = date(2024, 1, 1)
    manual_asset.purchase_price = Decimal("1.00")
    await session.commit()

    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    blocked = await handler(
        session=session,
        ctx=ctx,
        lots=[{"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00}],
        apply=True,
    )

    assert blocked["skipped_assets"] == 1
    assert blocked["updated_assets"] == 0
    assert blocked["inserted_values"] == 0
    await session.refresh(manual_asset)
    assert manual_asset.purchase_price == Decimal("1.00")

    allowed = await handler(
        session=session,
        ctx=ctx,
        lots=[{"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00}],
        apply=True,
        overwrite_purchase_fields=True,
    )
    assert allowed["updated_assets"] == 1
    await session.refresh(manual_asset)
    assert manual_asset.purchase_price == Decimal("100.00")


async def test_ambiguous_ticker_is_reported(
    session: AsyncSession,
    ctx: CallContext,
    test_user,
    test_workspace,
    manual_asset: Asset,
):
    session.add(
        Asset(
            id=uuid.uuid4(),
            user_id=test_user.id,
            workspace_id=test_workspace.id,
            name="AAPL34 segunda conta",
            type="investment",
            currency="BRL",
            valuation_method="manual",
            ticker="AAPL34",
            source="manual",
        )
    )
    await session.commit()

    handler = REGISTRY["propose_asset_purchase_backfill"].handler
    result = await handler(
        session=session,
        ctx=ctx,
        lots=[{"ticker": "AAPL34", "date": "2025-01-22", "amount": 100.00}],
        apply=True,
    )

    assert result["inserted_values"] == 0
    assert any("more than one" in error for error in result["errors"])
