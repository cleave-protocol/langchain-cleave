from __future__ import annotations

import pytest

from langchain_cleave import (
    CleaveAPIError,
    CleaveFindToken,
    CleaveGetGas,
    CleavePlanSwap,
    CleaveTrackSwap,
)

WALLET = "0x00000000000000000000000000000000000A11cE"
BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
BASE_WETH = "0x4200000000000000000000000000000000000006"
ARBITRUM_USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"


def test_plan_swap_usdc_to_weth_on_base() -> None:
    plan = CleavePlanSwap(handle_tool_error=False).invoke(
        {
            "fromChainId": 8453,
            "sellToken": "USDC",
            "buyToken": "WETH",
            "amount": "25",
            "wallet": WALLET,
        }
    )
    assert plan["blocked"] is False
    assert plan["requestId"]
    assert plan["route"]["outcomeKind"] in {"exact", "at_least"}
    assert plan["send"]["token"].lower() == BASE_USDC.lower()
    assert plan["receive"]["token"].lower() == BASE_WETH.lower()
    assert float(plan["receive"]["minimum"]) > 0
    assert plan["validUntil"] > plan["quotedAt"]
    assert plan["calls"]
    for c in plan["calls"]:
        assert c["chainId"] == 8453
        assert c["from"].lower() == WALLET.lower()
        assert c["to"].startswith("0x")
        assert c["data"].startswith("0x")


async def test_track_unknown_gas_order_is_not_found() -> None:
    tool = CleaveTrackSwap(handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke({"requestId": "g:8453:0x" + "0" * 63 + "1"})
    assert caught.value.status == 404
    assert caught.value.code == "not_found"
    assert caught.value.reason == "unknown_order"
    assert caught.value.retryable is False


def test_find_usdc_on_arbitrum() -> None:
    found = CleaveFindToken().invoke({"chainId": 42161, "query": "USDC"})
    assert found["ambiguous"] is False
    addresses = [c["address"].lower() for c in found["candidates"]]
    assert ARBITRUM_USDC.lower() in addresses
    usdc = found["candidates"][addresses.index(ARBITRUM_USDC.lower())]
    assert usdc["listed"] is True
    assert usdc["decimals"] == 6


def test_get_gas_offer_on_base() -> None:
    offer = CleaveGetGas(handle_tool_error=False).invoke(
        {"payChainId": 8453, "chainId": 8453, "recipient": WALLET, "amount": "5"}
    )
    assert offer["offer"]
    assert offer["payTo"].startswith("0x")
    assert offer["order"]["recipient"].lower() == WALLET.lower()
    assert float(offer["receive"]["minimum"]) > 0
    assert float(offer["fee"]["usdc"]) < 5
    assert offer["deadline"] > 0
