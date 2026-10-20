"""Tests for the asset proposal tools: creation and purchase backfill."""

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


# --- propose_create_asset --------------------------------------------------


async def test_create_asset_preview_does_not_write(
    session: AsyncSession, ctx: CallContext
):
    """A manual holding previews its value and cost basis and touches nothing."""
    handler = REGISTRY["propose_create_asset"].handler
    result = await handler(
        session=session,
        ctx=ctx,
        name="Corolla 2021",
        type="vehicle",
        currency="BRL",
        current_value=32000.0,
        purchase_date="2023-11-05",
        purchase_price=40000.0,
    )

    assert result["kind"] == "create_asset"
    assert result.get("applied") is not True
    assert result["errors"] == []
    assert result["name_collision"] is None
    assert result["proposed"]["name"] == "Corolla 2021"
    assert result["proposed"]["type"] == "vehicle"
    assert result["proposed"]["purchase_date"] == "2023-11-05"
    assert result["gain_loss_if_applied"] == -8000.0

    count = (await session.execute(select(func.count()).select_from(Asset))).scalar_one()
    assert count == 0


async def test_create_asset_flags_name_collision(
    session: AsyncSession, ctx: CallContext, test_user, test_workspace
):
    session.add(
        Asset(
            id=uuid.uuid4(),
            user_id=test_user.id,
            workspace_id=test_workspace.id,
            name="Corolla 2021",
            type="vehicle",
            currency="BRL",
            source="manual",
        )
    )
    await session.commit()

    handler = REGISTRY["propose_create_asset"].handler
    result = await handler(
        session=session, ctx=ctx, name="corolla 2021", type="vehicle"
    )

    assert result["name_collision"] is not None
    assert result["name_collision"]["name"] == "Corolla 2021"


async def test_create_asset_refuses_collision_on_apply(
    session: AsyncSession, ctx: CallContext, test_user, test_workspace
):
    session.add(
        Asset(
            id=uuid.uuid4(),
            user_id=test_user.id,
            workspace_id=test_workspace.id,
            name="Corolla 2021",
            type="vehicle",
            currency="BRL",
            source="manual",
        )
    )
    await session.commit()
    before = (await session.execute(select(func.count()).select_from(Asset))).scalar_one()

    handler = REGISTRY["propose_create_asset"].handler
    result = await handler(
        session=session, ctx=ctx, name="Corolla 2021", type="vehicle", apply=True
    )

    assert result.get("applied") is not True
    assert "already exists" in result["error"]
    after = (await session.execute(select(func.count()).select_from(Asset))).scalar_one()
    assert after == before


async def test_create_asset_reports_invalid_combination(
    session: AsyncSession, ctx: CallContext
):
    handler = REGISTRY["propose_create_asset"].handler
    result = await handler(
        session=session,
        ctx=ctx,
        name="Corolla 2021",
        type="vehicle",
        sell_price=25000.0,
    )

    assert any("sell_date is required" in error for error in result["errors"])
    assert result.get("applied") is not True


async def test_create_asset_rejects_unknown_type(
    session: AsyncSession, ctx: CallContext
):
    handler = REGISTRY["propose_create_asset"].handler
    result = await handler(session=session, ctx=ctx, name="Drone", type="gadget")

    assert any("type must be one of" in error for error in result["errors"])


async def test_create_asset_apply_writes_row_and_value(
    session: AsyncSession, ctx: CallContext
):
    """The row keeps both the current value and the purchase cost, which is
    what makes a later sale measurable as gain/loss."""
    handler = REGISTRY["propose_create_asset"].handler

    result = await handler(
        session=session,
        ctx=ctx,
        name="Corolla 2021",
        type="vehicle",
        currency="BRL",
        current_value=32000.0,
        purchase_date="2023-11-05",
        purchase_price=40000.0,
        apply=True,
    )

    assert result.get("applied") is True
    assert result["gain_loss"] == -8000.0

    asset = (
        await session.execute(
            select(Asset).where(Asset.id == uuid.UUID(result["id"]))
        )
    ).scalar_one()
    assert asset.name == "Corolla 2021"
    assert asset.type == "vehicle"
    assert asset.currency == "BRL"
    assert asset.source == "manual"
    purchase_price = asset.purchase_price
    purchase_date = asset.purchase_date
    assert purchase_price is not None
    assert purchase_date is not None
    assert float(purchase_price) == 40000.0
    assert purchase_date == date(2023, 11, 5)

    values = (
        (
            await session.execute(
                select(AssetValue).where(AssetValue.asset_id == asset.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(values) == 1
    assert float(values[0].amount) == 32000.0


async def test_preview_archive_does_not_write(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    handler = REGISTRY["propose_archive_assets"].handler
    asset_id = manual_asset.id
    result = await handler(
        session=session,
        ctx=ctx,
        asset_ids=[str(asset_id)],
    )

    assert result["kind"] == "archive_assets"
    assert result["is_archived"] is True
    assert result["assets"][0]["id"] == str(asset_id)
    assert result["assets"][0]["is_archived"] is False

    session.expire_all()
    fresh = (
        await session.execute(select(Asset).where(Asset.id == asset_id))
    ).scalar_one()
    assert fresh is not None
    assert fresh.is_archived is False


async def test_apply_archive_writes(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    handler = REGISTRY["propose_archive_assets"].handler
    asset_id = manual_asset.id
    result = await handler(
        session=session,
        ctx=ctx,
        asset_ids=[str(asset_id)],
        apply=True,
    )

    assert result["applied"] is True
    assert result["updated_count"] == 1

    session.expire_all()
    fresh = (
        await session.execute(select(Asset).where(Asset.id == asset_id))
    ).scalar_one()
    assert fresh is not None
    assert fresh.is_archived is True


async def test_preview_update_asset_does_not_write(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    handler = REGISTRY["propose_update_asset"].handler
    asset_id = manual_asset.id
    result = await handler(
        session=session,
        ctx=ctx,
        asset_id=str(asset_id),
        name="Hyundai i30 2011",
    )

    assert result["kind"] == "update_asset"
    assert result["target"]["name"] == "AAPL34"
    assert result["changes"]["name"] == "Hyundai i30 2011"

    session.expire_all()
    fresh = (
        await session.execute(select(Asset).where(Asset.id == asset_id))
    ).scalar_one()
    assert fresh.name == "AAPL34"


async def test_apply_update_asset_writes_name_and_value(
    session: AsyncSession, ctx: CallContext, manual_asset: Asset
):
    handler = REGISTRY["propose_update_asset"].handler
    asset_id = manual_asset.id
    result = await handler(
        session=session,
        ctx=ctx,
        asset_id=str(asset_id),
        name="Hyundai i30 2011",
        current_value=32000,
        current_value_date="2023-11-05",
        apply=True,
    )

    assert result["applied"] is True
    assert result["current_value_added"] is True

    session.expire_all()
    fresh = (
        await session.execute(select(Asset).where(Asset.id == asset_id))
    ).scalar_one()
    assert fresh.name == "Hyundai i30 2011"
    values = (
        (
            await session.execute(
                select(AssetValue).where(AssetValue.asset_id == asset_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(values) == 1
    assert float(values[0].amount) == 32000.0
    assert values[0].date == date(2023, 11, 5)
