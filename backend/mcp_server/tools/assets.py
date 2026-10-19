"""Asset purchase backfill.

Manual assets created from a position snapshot have a current value but no
purchase history. Net-worth reports before that snapshot then treat the
holding as zero. This tool previews and applies a purchase-lot history for
those holdings: it sets ``purchase_date`` and ``purchase_price`` and inserts
one manual ``asset_values`` row per purchase date with the cumulative amount
invested up to that date.

Lots can be passed explicitly, for statement data that is not in Securo yet,
or derived from debit transactions whose description starts with a known
prefix (``Compra de BDR - `` by default).

The tool refuses market-priced and ledger-backed assets: their position and
cost basis come from ``asset_transactions`` and are recomputed from that
ledger, so writing ``purchase_price`` directly would be overwritten. It also
refuses provider-synced assets unless ``force=true``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.asset import Asset
from app.models.asset_transaction import AssetTransaction
from app.models.asset_value import AssetValue
from app.models.transaction import Transaction
from app.services.fx_rate_service import stamp_primary_amount
from mcp_server.auth import CallContext
from mcp_server.registry import tool
from mcp_server.tools._helpers import parse_date, parse_uuid, resolve_workspace_id
from mcp_server.tools.proposals import _APPLY_FIELD, _PROPOSAL_PREFACE, _can_apply


_LOT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "minLength": 1},
        "date": {"type": "string", "format": "date"},
        "amount": {"type": "number", "exclusiveMinimum": 0},
        "currency": {"type": "string", "minLength": 3, "maxLength": 3},
        "asset_id": {"type": "string", "format": "uuid"},
    },
    "required": ["ticker", "date", "amount"],
    "additionalProperties": False,
}

_PARAMETERS = {
    "type": "object",
    "properties": {
        "lots": {
            "type": "array",
            "items": _LOT_SCHEMA,
            "description": (
                "Explicit purchase lots. Use this when the statement is not in "
                "Securo. Every amount is the cash invested on that date, in the "
                "asset currency. Omit to derive lots from transactions."
            ),
        },
        "description_prefix": {
            "type": "string",
            "default": "Compra de BDR - ",
            "description": (
                "When lots is omitted, only debit transactions whose description "
                "starts with this prefix are used. The ticker is the first word "
                "after the prefix."
            ),
        },
        "from_date": {"type": "string", "format": "date"},
        "to_date": {"type": "string", "format": "date"},
        "overwrite_purchase_fields": {
            "type": "boolean",
            "default": False,
            "description": (
                "Replace purchase_date and purchase_price when they are already "
                "set and differ from the proposed values."
            ),
        },
        "force": {
            "type": "boolean",
            "default": False,
            "description": (
                "Allow a provider-synced asset. Ledger-backed and market-priced "
                "assets are still refused."
            ),
        },
        "apply": _APPLY_FIELD,
    },
    "additionalProperties": False,
}


def _parse_lot(raw: Any, index: int, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        errors.append(f"lot {index}: expected an object")
        return None

    ticker = str(raw.get("ticker") or "").strip().upper()
    if not ticker:
        errors.append(f"lot {index}: ticker is required")
        return None

    try:
        lot_date = parse_date(raw.get("date"))
    except ValueError:
        errors.append(f"lot {index}: date is not a valid ISO date")
        return None
    if lot_date is None:
        errors.append(f"lot {index}: date is required")
        return None

    try:
        amount = Decimal(str(raw.get("amount")))
    except (InvalidOperation, TypeError, ValueError):
        errors.append(f"lot {index}: amount is not a number")
        return None
    if amount <= 0:
        errors.append(f"lot {index}: amount must be greater than zero")
        return None

    currency = raw.get("currency")
    if currency is not None:
        currency = str(currency).strip().upper()
        if len(currency) != 3:
            errors.append(f"lot {index}: currency must be a three-letter code")
            return None

    asset_id = None
    if raw.get("asset_id"):
        try:
            asset_id = parse_uuid(raw.get("asset_id"))
        except (TypeError, ValueError):
            errors.append(f"lot {index}: asset_id is not a UUID")
            return None

    return {
        "ticker": ticker,
        "date": lot_date,
        "amount": amount,
        "currency": currency,
        "asset_id": asset_id,
    }


async def _lots_from_transactions(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    prefix: str,
    start: date | None,
    end: date | None,
    errors: list[str],
) -> list[dict[str, Any]]:
    prefix = prefix.strip()
    if not prefix:
        errors.append("description_prefix must not be blank")
        return []

    stmt = select(Transaction).where(
        Transaction.workspace_id == workspace_id,
        Transaction.type == "debit",
        Transaction.description.ilike(f"{prefix}%"),
    )
    if start is not None:
        stmt = stmt.where(Transaction.date >= start)
    if end is not None:
        stmt = stmt.where(Transaction.date <= end)

    rows = (await session.execute(stmt.order_by(Transaction.date, Transaction.id))).scalars().all()

    lots: list[dict[str, Any]] = []
    for tx in rows:
        remainder = tx.description[len(prefix) :].strip()
        ticker = remainder.split()[0].strip(" .,;:-").upper() if remainder else ""
        if not ticker:
            continue
        amount = abs(Decimal(str(tx.amount)))
        if amount <= 0:
            continue
        lots.append(
            {
                "ticker": ticker,
                "date": tx.date,
                "amount": amount,
                "currency": tx.currency.upper() if tx.currency else None,
                "asset_id": None,
            }
        )

    if not lots:
        errors.append(f"no debit transactions found with description prefix {prefix!r}")
    return lots


async def _resolve_asset(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    lot: dict[str, Any],
) -> tuple[Asset | None, str | None]:
    if lot["asset_id"] is not None:
        asset = await session.get(Asset, lot["asset_id"])
        if asset is None or asset.workspace_id != workspace_id:
            return None, f"asset {lot['asset_id']} not found in this workspace"
        if asset.ticker and asset.ticker.upper() != lot["ticker"]:
            return (
                None,
                f"asset {asset.name!r} has ticker {asset.ticker!r}, not {lot['ticker']!r}",
            )
        return asset, None

    ticker = lot["ticker"]
    rows = (
        (
            await session.execute(
                select(Asset).where(
                    Asset.workspace_id == workspace_id,
                    or_(
                        func.upper(Asset.ticker) == ticker,
                        func.upper(Asset.name) == ticker,
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return (
            None,
            f"no asset found for ticker {ticker!r}; pass asset_id in the lot "
            "to target a holding without a matching ticker",
        )
    if len(rows) > 1:
        names = ", ".join(f"{asset.name!r} ({asset.id})" for asset in rows)
        return (
            None,
            f"ticker {ticker!r} matches more than one asset: {names}; pass asset_id",
        )
    return rows[0], None


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


@tool(
    name="propose_asset_purchase_backfill",
    description=_PROPOSAL_PREFACE
    + (
        "Preview and apply purchase history for a manual asset that only has a "
        "current value. Pass lots as [{ticker, date, amount}] when the "
        "purchases come from a statement that is not in Securo yet; omit lots "
        "to derive them from debit transactions whose description starts with "
        "description_prefix (default 'Compra de BDR - '). The tool sets the "
        "asset's purchase_date and purchase_price and inserts one manual "
        "asset_values row per purchase date with the cumulative invested "
        "amount. It refuses market-priced and ledger-backed assets, and "
        "provider-synced assets unless force=true. Run it once without apply "
        "to review the diff."
    ),
    parameters=_PARAMETERS,
    is_proposal=True,
    tags=["propose", "assets"],
)
async def propose_asset_purchase_backfill(
    *,
    session: AsyncSession,
    ctx: CallContext,
    lots: list[dict[str, Any]] | None = None,
    description_prefix: str = "Compra de BDR - ",
    from_date: str | None = None,
    to_date: str | None = None,
    overwrite_purchase_fields: bool = False,
    force: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    ws_id = await resolve_workspace_id(session, ctx)
    errors: list[str] = []

    raw_lots: list[dict[str, Any]] = []
    if lots is not None:
        if not lots:
            errors.append("lots was empty; omit it to derive lots from transactions")
        for index, raw in enumerate(lots):
            parsed = _parse_lot(raw, index, errors)
            if parsed is not None:
                raw_lots.append(parsed)
    else:
        try:
            start = parse_date(from_date)
            end = parse_date(to_date)
        except ValueError:
            errors.append("from_date and to_date must be valid ISO dates")
            raw_lots = []
        else:
            raw_lots = await _lots_from_transactions(
                session,
                ws_id,
                description_prefix,
                start,
                end,
                errors,
            )

    by_asset: dict[uuid.UUID, list[dict[str, Any]]] = defaultdict(list)
    assets: dict[uuid.UUID, Asset] = {}
    for lot in raw_lots:
        asset, error = await _resolve_asset(session, ws_id, lot)
        if error is not None:
            errors.append(error)
            continue
        assert asset is not None
        assets[asset.id] = asset
        by_asset[asset.id].append(lot)

    items: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for asset_id, asset_lots in by_asset.items():
        asset = assets[asset_id]
        label = asset.ticker or asset.name

        if asset.valuation_method == "market_price":
            errors.append(
                f"{label}: market-priced assets derive their history from the "
                "asset transaction ledger; use the asset order import instead"
            )
            continue
        if asset.average_price is not None:
            errors.append(
                f"{label}: ledger-backed (average_price is set); the asset "
                "transaction ledger is authoritative"
            )
            continue

        tx_count = (
            await session.scalar(
                select(func.count())
                .select_from(AssetTransaction)
                .where(AssetTransaction.asset_id == asset.id)
            )
            or 0
        )
        if tx_count:
            errors.append(
                f"{label}: ledger-backed ({tx_count} asset transactions); the "
                "asset transaction ledger is authoritative"
            )
            continue
        if asset.source != "manual" and not force:
            errors.append(
                f"{label}: source is {asset.source!r}; pass force=true to "
                "backfill a provider-synced asset"
            )
            continue

        lots_by_date: dict[date, Decimal] = {}
        currency_error = False
        for lot in asset_lots:
            if lot["currency"] and lot["currency"] != asset.currency.upper():
                errors.append(
                    f"{label}: lot on {lot['date'].isoformat()} is in "
                    f"{lot['currency']}, asset is in {asset.currency}; the tool "
                    "does not convert currencies"
                )
                currency_error = True
                continue
            lots_by_date[lot["date"]] = lots_by_date.get(lot["date"], Decimal("0")) + lot["amount"]
        if currency_error or not lots_by_date:
            continue

        dates = sorted(lots_by_date)
        proposed_date = dates[0]
        proposed_price = _money(sum(lots_by_date.values()))

        existing_rows = (
            (
                await session.execute(
                    select(AssetValue).where(
                        AssetValue.asset_id == asset.id,
                        AssetValue.date.in_(dates),
                    )
                )
            )
            .scalars()
            .all()
        )
        existing_by_date: dict[date, list[AssetValue]] = defaultdict(list)
        for row in existing_rows:
            existing_by_date[row.date].append(row)

        latest = (
            await session.execute(
                select(AssetValue)
                .where(AssetValue.asset_id == asset.id)
                .order_by(AssetValue.date.desc(), AssetValue.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        current_price = _money(asset.purchase_price) if asset.purchase_price is not None else None
        if asset.purchase_date == proposed_date and current_price == proposed_price:
            purchase_action = "unchanged"
        elif asset.purchase_date is None and current_price is None:
            purchase_action = "set"
        elif overwrite_purchase_fields:
            purchase_action = "overwrite"
        else:
            purchase_action = "conflict"

        warnings: list[str] = []
        if purchase_action == "conflict":
            warnings.append(
                "purchase_date or purchase_price is already set and differs; "
                "pass overwrite_purchase_fields=true to replace them"
            )

        cumulative = Decimal("0")
        value_points: list[dict[str, Any]] = []
        insert_dates: list[date] = []
        skipped_values = 0
        for value_date in dates:
            cumulative += lots_by_date[value_date]
            existing = existing_by_date.get(value_date, [])
            action = "skip" if existing else "insert"
            if existing:
                skipped_values += 1
            else:
                insert_dates.append(value_date)
            value_points.append(
                {
                    "date": value_date.isoformat(),
                    "amount": float(_money(cumulative)),
                    "action": action,
                    "existing": [
                        {
                            "amount": float(_money(Decimal(str(row.amount)))),
                            "source": row.source,
                        }
                        for row in existing
                    ]
                    or None,
                }
            )

        items.append(
            {
                "asset_id": str(asset.id),
                "name": asset.name,
                "ticker": asset.ticker,
                "currency": asset.currency,
                "source": asset.source,
                "current_purchase_date": (
                    asset.purchase_date.isoformat() if asset.purchase_date else None
                ),
                "current_purchase_price": (
                    float(current_price) if current_price is not None else None
                ),
                "proposed_purchase_date": proposed_date.isoformat(),
                "proposed_purchase_price": float(proposed_price),
                "purchase_action": purchase_action,
                "lot_count": len(dates),
                "value_points": value_points,
                "current_value": (
                    {
                        "date": latest.date.isoformat(),
                        "amount": float(_money(Decimal(str(latest.amount)))),
                        "source": latest.source,
                    }
                    if latest is not None
                    else None
                ),
                "warnings": warnings,
            }
        )
        pending.append(
            {
                "asset": asset,
                "purchase_action": purchase_action,
                "proposed_date": proposed_date,
                "proposed_price": proposed_price,
                "lots_by_date": lots_by_date,
                "insert_dates": insert_dates,
                "skipped_values": skipped_values,
            }
        )

    updated_assets = 0
    inserted_values = 0
    skipped_values = sum(item["skipped_values"] for item in pending)
    skipped_assets = 0
    can_apply = _can_apply(ctx, apply)

    if can_apply:
        for item in pending:
            if item["purchase_action"] == "conflict":
                skipped_assets += 1
                continue
            asset = item["asset"]
            if item["purchase_action"] in ("set", "overwrite"):
                asset.purchase_date = item["proposed_date"]
                asset.purchase_price = item["proposed_price"]
                await stamp_primary_amount(
                    session,
                    ctx.user_id,
                    asset,
                    amount_field="purchase_price",
                    primary_field="purchase_price_primary",
                    rate_field="_no_rate",
                    date_field="purchase_date",
                )
                updated_assets += 1

            cumulative = Decimal("0")
            for value_date in sorted(item["lots_by_date"]):
                cumulative += item["lots_by_date"][value_date]
                if value_date not in item["insert_dates"]:
                    continue
                session.add(
                    AssetValue(
                        asset_id=asset.id,
                        workspace_id=ws_id,
                        amount=_money(cumulative),
                        date=value_date,
                        source="manual",
                    )
                )
                inserted_values += 1

        if updated_assets or inserted_values:
            await session.commit()

    applied = bool(can_apply and (updated_assets or inserted_values))
    return {
        "kind": "asset_purchase_backfill",
        "items": items,
        "errors": errors,
        "applied": applied,
        "updated_assets": updated_assets,
        "inserted_values": inserted_values,
        "skipped_values": skipped_values,
        "skipped_assets": skipped_assets,
    }
