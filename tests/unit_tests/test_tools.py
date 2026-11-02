from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ValidationError

from langchain_cleave import (
    CleaveAPIError,
    CleaveAPIWrapper,
    CleaveFindToken,
    CleaveGetGas,
    CleavePlanSwap,
    CleaveToolkit,
    CleaveTrackSwap,
    FindTokenInput,
    GetGasInput,
    PlanSwapInput,
    TrackSwapInput,
    _descriptions,
)
from langchain_cleave.tools import (
    GAS_SUBMIT,
    OFFERS_KEPT,
    PLAN_CONFIRM,
    RELAY_OFF,
)

WALLET = "0x00000000000000000000000000000000000A11cE"
GAS_ID = "g:8453:0x" + "0" * 63 + "1"
TX = "0x" + "ab" * 32

PLAN_BODY = {
    "requestId": "75a208eb-cba3-4fd4-ba17-d0eabbfd73b6",
    "blocked": False,
    "requiresConfirmation": True,
    "route": {"class": "same_chain", "outcomeKind": "at_least"},
    "receive": {"symbol": "WETH", "expected": "0.0109", "minimum": "0.0108"},
    "fees": [],
    "notices": [],
    "validUntil": 1793623133,
    "calls": [
        {
            "chainId": 8453,
            "from": WALLET.lower(),
            "to": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "data": "0x095ea7b3",
            "value": "0x0",
        }
    ],
}
TRACK_BODY = {
    "status": "delivered",
    "matchesPlan": True,
    "arrived": None,
    "planned": None,
    "next": "Delivered.",
    "pollAfterSec": None,
    "receipt": None,
}
FIND_BODY = {
    "ambiguous": False,
    "candidates": [
        {
            "address": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            "decimals": 6,
            "listed": True,
            "symbol": "USDC",
            "name": "USD Coin",
            "flags": [],
        }
    ],
}
GAS_BODY = {
    "offer": "g2.example",
    "kind": "swap",
    "payChainId": 8453,
    "chainId": 8453,
    "payTo": "0x245eCD38d796641411A242dE1Cb72e1591de1106",
    "fee": {"usdc": "0.041258"},
    "receive": {"symbol": "ETH", "minimum": "0.0021"},
    "deadline": 1793623969,
}

CASES = [
    pytest.param(
        CleavePlanSwap,
        "/agent/v1/plan_swap",
        {
            "fromChainId": 8453,
            "sellToken": "USDC",
            "buyToken": "WETH",
            "amount": "25",
            "wallet": WALLET,
        },
        PLAN_BODY,
        id="plan_swap",
    ),
    pytest.param(
        CleaveTrackSwap,
        "/agent/v1/track_swap",
        {"chainId": 8453, "txHash": TX, "requestId": "75a208eb-cba3"},
        TRACK_BODY,
        id="track_swap",
    ),
    pytest.param(
        CleaveFindToken,
        "/agent/v1/find_token",
        {"chainId": 42161, "query": "USDC"},
        FIND_BODY,
        id="find_token",
    ),
    pytest.param(
        CleaveGetGas,
        "/agent/v1/get_gas",
        {"payChainId": 8453, "chainId": 8453, "recipient": WALLET, "amount": "5"},
        GAS_BODY,
        id="get_gas",
    ),
]

PLAN_ARGS = {
    "fromChainId": 8453,
    "sellToken": "USDC",
    "buyToken": "WETH",
    "amount": "25",
    "wallet": WALLET,
}

BAD_ARGS = [
    pytest.param(
        CleavePlanSwap,
        {
            "fromChainId": 999,
            "sellToken": "USDC",
            "buyToken": "WETH",
            "amount": "25",
            "wallet": WALLET,
        },
        id="plan_swap-chain",
    ),
    pytest.param(
        CleavePlanSwap,
        {
            "fromChainId": 8453,
            "sellToken": "USDC",
            "buyToken": "WETH",
            "amount": "25",
            "wallet": WALLET,
            "maxSlippageBps": 5001,
        },
        id="plan_swap-slippage",
    ),
    pytest.param(
        CleavePlanSwap,
        {
            "fromChainId": 8453,
            "sellToken": "USDC",
            "buyToken": "WETH",
            "amount": "1e3",
            "wallet": WALLET,
        },
        id="plan_swap-amount",
    ),
    pytest.param(
        CleavePlanSwap,
        {"fromChainId": 8453, "sellToken": "USDC", "amount": "1", "wallet": WALLET},
        id="plan_swap-missing",
    ),
    pytest.param(
        CleaveTrackSwap,
        {"chainId": 8453, "txHash": "0x1234"},
        id="track_swap-hash",
    ),
    pytest.param(
        CleaveTrackSwap,
        {"requestId": "bad id!"},
        id="track_swap-request-id",
    ),
    pytest.param(
        CleaveFindToken,
        {"chainId": 42161, "query": ""},
        id="find_token-query",
    ),
    pytest.param(
        CleaveFindToken,
        {"chainId": 42161, "query": "USDC", "limit": 5},
        id="find_token-extra",
    ),
    pytest.param(
        CleaveGetGas,
        {"payChainId": 8453, "chainId": 8453, "recipient": "0x12", "amount": "5"},
        id="get_gas-recipient",
    ),
    pytest.param(
        CleaveGetGas,
        {"offer": "g2.x", "signature": "0xzz", "wallet": WALLET},
        id="get_gas-signature",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "fromChainId": True},
        id="plan_swap-chain-bool",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "toChainId": False},
        id="plan_swap-to-chain-bool",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "fromChainId": "8453"},
        id="plan_swap-chain-string",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "maxSlippageBps": True},
        id="plan_swap-slippage-bool",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "maxSlippageBps": "50"},
        id="plan_swap-slippage-string",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "maxSlippageBps": 50.0},
        id="plan_swap-slippage-float",
    ),
    pytest.param(
        CleavePlanSwap,
        {**PLAN_ARGS, "maxSlippageBps": -1},
        id="plan_swap-slippage-negative",
    ),
    pytest.param(
        CleaveTrackSwap,
        {"chainId": True, "txHash": TX},
        id="track_swap-chain-bool",
    ),
    pytest.param(
        CleaveFindToken,
        {"chainId": 42161, "query": "U" * 65},
        id="find_token-query-long",
    ),
    pytest.param(
        CleaveGetGas,
        {"offer": "g" * 4097, "signature": "0x1b", "wallet": WALLET},
        id="get_gas-offer-long",
    ),
    pytest.param(
        CleaveGetGas,
        {"payChainId": True, "chainId": 8453, "recipient": WALLET, "amount": "5"},
        id="get_gas-chain-bool",
    ),
]

REFUSAL = {
    "apiVersion": "1",
    "requestId": "9a171b3f-2bcf-40a2-9d07-2abf804e0b92",
    "error": {
        "code": "not_found",
        "reason": "unknown_order",
        "message": "This part of the service does not exist.",
        "detail": "No gas order has this requestId",
        "retryable": False,
    },
}

Handler = Callable[[httpx.Request], httpx.Response]


def mocked(
    handler: Handler, **kwargs: Any
) -> tuple[CleaveAPIWrapper, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(record)
    api = CleaveAPIWrapper(
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )
    return api, seen


def answer(status: int, body: Any) -> Handler:
    return lambda request: httpx.Response(status, json=body)


def call(tool: BaseTool, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": tool.name, "args": args, "id": "call-1", "type": "tool_call"}


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLEAVE_API_KEY", raising=False)


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
def test_success(cls: type[BaseTool], path: str, args: dict, body: dict) -> None:
    api, seen = mocked(answer(200, body))
    tool = cls(api_wrapper=api)
    assert tool.invoke(args) == body
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.cleave.fi" + path
    assert json.loads(request.content) == args
    assert request.headers["content-type"] == "application/json"
    assert request.headers["user-agent"].startswith("langchain-cleave/")
    assert "authorization" not in request.headers
    assert "origin" not in request.headers
    assert request.url.query == b""


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
async def test_success_async(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    api, seen = mocked(answer(200, body))
    tool = cls(api_wrapper=api)
    assert await tool.ainvoke(args) == body
    assert len(seen) == 1
    assert seen[0].url.path == path
    assert json.loads(seen[0].content) == args


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
def test_tool_call_returns_json(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    api, _ = mocked(answer(200, body))
    tool = cls(api_wrapper=api)
    message = tool.invoke(call(tool, args))
    assert isinstance(message, ToolMessage)
    assert message.status == "success"
    assert message.tool_call_id == "call-1"
    assert json.loads(str(message.content)) == body


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
def test_api_error_is_tool_content(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    api, seen = mocked(answer(404, REFUSAL))
    tool = cls(api_wrapper=api)
    message = tool.invoke(call(tool, args))
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert json.loads(str(message.content)) == REFUSAL
    assert json.loads(tool.invoke(args)) == REFUSAL
    assert len(seen) == 2


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
async def test_api_error_is_tool_content_async(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    api, _ = mocked(answer(400, REFUSAL))
    tool = cls(api_wrapper=api)
    message = await tool.ainvoke(call(tool, args))
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert json.loads(str(message.content))["error"]["code"] == "not_found"


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
def test_api_error_raises_when_unhandled(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    api, _ = mocked(answer(404, REFUSAL))
    tool = cls(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        tool.invoke(args)
    assert caught.value.status == 404
    assert caught.value.code == "not_found"
    assert caught.value.reason == "unknown_order"
    assert caught.value.retryable is False
    assert caught.value.body == REFUSAL


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
async def test_api_error_raises_when_unhandled_async(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    limited = {
        "apiVersion": "1",
        "requestId": "r",
        "error": {
            "code": "rate_limited",
            "reason": "agent_limit",
            "message": "Busy",
            "retryable": True,
            "retryAfterSec": 2,
        },
    }
    api, _ = mocked(answer(429, limited))
    tool = cls(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(args)
    assert caught.value.status == 429
    assert caught.value.retryable is True
    assert caught.value.body["error"]["retryAfterSec"] == 2


def invalid_param(content: Any) -> dict[str, Any]:
    error = json.loads(str(content))["error"]
    assert error["code"] == "invalid_param"
    assert error["message"] == "Some details of this request are not valid."
    assert error["detail"]
    assert error["retryable"] is False
    return error


@pytest.mark.parametrize(("cls", "args"), BAD_ARGS)
def test_validation_is_invalid_param_content_before_http(
    cls: type[BaseTool], args: dict
) -> None:
    api, seen = mocked(answer(200, {}))
    tool = cls(api_wrapper=api)
    message = tool.invoke(call(tool, args))
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    invalid_param(message.content)
    invalid_param(tool.invoke(args))
    assert seen == []


@pytest.mark.parametrize(("cls", "args"), BAD_ARGS)
async def test_validation_is_invalid_param_content_before_http_async(
    cls: type[BaseTool], args: dict
) -> None:
    api, seen = mocked(answer(200, {}))
    tool = cls(api_wrapper=api)
    message = await tool.ainvoke(call(tool, args))
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    invalid_param(message.content)
    assert seen == []


@pytest.mark.parametrize(("cls", "args"), BAD_ARGS)
def test_validation_raises_when_unhandled(cls: type[BaseTool], args: dict) -> None:
    api, seen = mocked(answer(200, {}))
    tool = cls(api_wrapper=api, handle_validation_error=False)
    with pytest.raises(ValidationError):
        tool.invoke(args)
    assert seen == []


def test_invalid_param_detail_names_the_field() -> None:
    api, _ = mocked(answer(200, {}))
    error = invalid_param(
        CleavePlanSwap(api_wrapper=api).invoke({**PLAN_ARGS, "maxSlippageBps": 6000})
    )
    assert "maxSlippageBps" in error["detail"]


def test_boundary_values_are_accepted() -> None:
    api, seen = mocked(answer(200, {}))
    CleavePlanSwap(api_wrapper=api).invoke({**PLAN_ARGS, "maxSlippageBps": 0})
    CleavePlanSwap(api_wrapper=api).invoke({**PLAN_ARGS, "maxSlippageBps": 5000})
    CleaveFindToken(api_wrapper=api).invoke({"chainId": 1, "query": "U" * 64})
    CleaveGetGas(api_wrapper=api, allow_relay=True).invoke(
        {"offer": "g" * 4096, "signature": "0x1b", "wallet": WALLET}
    )
    CleaveTrackSwap(api_wrapper=api).invoke({"requestId": "a" * 64})
    assert [json.loads(r.content).get("maxSlippageBps") for r in seen[:2]] == [0, 5000]
    assert len(seen) == 5


@pytest.mark.parametrize(("cls", "path", "args", "body"), CASES)
def test_network_error_is_tool_content(
    cls: type[BaseTool], path: str, args: dict, body: dict
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    api, _ = mocked(fail)
    tool = cls(api_wrapper=api)
    message = tool.invoke(call(tool, args))
    assert message.status == "error"
    error = json.loads(str(message.content))["error"]
    assert error["code"] == "network_error"
    assert error["retryable"] is True


def test_non_json_error_is_http_error() -> None:
    api, _ = mocked(
        lambda request: httpx.Response(502, text="<html>bad gateway</html>")
    )
    tool = CleaveFindToken(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        tool.invoke({"chainId": 1, "query": "USDC"})
    assert caught.value.status == 502
    assert caught.value.code == "http_error"
    assert caught.value.retryable is True


def test_success_that_is_not_an_object_is_refused() -> None:
    api, _ = mocked(answer(200, ["not", "an", "object"]))
    tool = CleaveFindToken(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        tool.invoke({"chainId": 1, "query": "USDC"})
    assert caught.value.code == "bad_response"


OPTIONAL_FIELDS = [
    (CleavePlanSwap, PLAN_ARGS, "toChainId"),
    (CleavePlanSwap, PLAN_ARGS, "recipient"),
    (CleavePlanSwap, PLAN_ARGS, "maxSlippageBps"),
    (CleaveTrackSwap, {"requestId": GAS_ID}, "chainId"),
    (CleaveTrackSwap, {"requestId": GAS_ID}, "txHash"),
    (CleaveTrackSwap, {"chainId": 8453, "txHash": TX}, "requestId"),
    (
        CleaveGetGas,
        {"payChainId": 8453, "chainId": 8453, "recipient": WALLET, "amount": "5"},
        "wallet",
    ),
    (
        CleaveGetGas,
        {"payChainId": 8453, "chainId": 8453, "recipient": WALLET, "amount": "5"},
        "refundTo",
    ),
    (CleaveGetGas, {"chainId": 8453, "recipient": WALLET, "amount": "5"}, "payChainId"),
]


@pytest.mark.parametrize(
    ("cls", "base", "field"),
    OPTIONAL_FIELDS,
    ids=[f"{c.__name__}-{f}" for c, _, f in OPTIONAL_FIELDS],
)
def test_explicit_null_is_invalid_param(
    cls: type[BaseTool], base: dict, field: str
) -> None:
    api, seen = mocked(answer(200, {}))
    tool = cls(api_wrapper=api)
    error = invalid_param(tool.invoke({**base, field: None}))
    assert field in error["detail"]
    assert seen == []


def test_omitted_fields_are_not_sent() -> None:
    api, seen = mocked(answer(200, PLAN_BODY))
    CleavePlanSwap(api_wrapper=api).invoke(PLAN_ARGS)
    assert json.loads(seen[0].content) == PLAN_ARGS


def test_cross_chain_plan_sends_every_field() -> None:
    api, seen = mocked(answer(200, PLAN_BODY))
    args = {
        "fromChainId": 42161,
        "toChainId": 10,
        "sellToken": "USDC",
        "buyToken": "ETH",
        "amount": "100.5",
        "wallet": WALLET,
        "recipient": "0x" + "1" * 40,
        "maxSlippageBps": 100,
    }
    CleavePlanSwap(api_wrapper=api).invoke(args)
    assert json.loads(seen[0].content) == args


def test_track_by_gas_request_id_alone() -> None:
    api, seen = mocked(answer(200, TRACK_BODY))
    CleaveTrackSwap(api_wrapper=api).invoke({"requestId": GAS_ID})
    assert json.loads(seen[0].content) == {"requestId": GAS_ID}


SIGNED = {"offer": "g2.abc.def", "signature": "0x" + "1b" * 65, "wallet": WALLET}


def test_get_gas_submit_passes_signed_offer_when_relay_allowed() -> None:
    api, seen = mocked(answer(200, {"requestId": GAS_ID}))
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True)
    assert tool.invoke(SIGNED) == {"requestId": GAS_ID}
    assert json.loads(seen[0].content) == SIGNED
    assert tool.description == CleaveGetGas.model_fields["description"].default


@pytest.mark.parametrize(
    "args",
    [SIGNED, {"offer": "g2.abc.def", "wallet": WALLET}],
    ids=["offer-and-signature", "offer-only"],
)
def test_get_gas_relay_is_off_by_default(args: dict) -> None:
    api, seen = mocked(answer(200, {"requestId": GAS_ID}))
    tool = CleaveGetGas(api_wrapper=api)
    assert tool.allow_relay is False
    assert "turned off" in tool.description
    message = tool.invoke(call(tool, args))
    assert message.status == "error"
    assert json.loads(message.content)["error"]["code"] == "relay_disabled"
    assert seen == []


async def test_get_gas_relay_is_off_by_default_async() -> None:
    api, seen = mocked(answer(200, {"requestId": GAS_ID}))
    tool = CleaveGetGas(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(SIGNED)
    assert caught.value.code == "relay_disabled"
    assert seen == []


def test_get_gas_offer_works_with_relay_off() -> None:
    api, seen = mocked(answer(200, GAS_BODY))
    args = {"payChainId": 8453, "chainId": 10, "recipient": WALLET, "amount": "5"}
    assert CleaveGetGas(api_wrapper=api).invoke(args) == GAS_BODY
    assert json.loads(seen[0].content) == args


def test_toolkit_gas_relay_option() -> None:
    api, seen = mocked(answer(200, {"requestId": GAS_ID}))
    off = CleaveToolkit(api_wrapper=api).get_tools()[3]
    on = CleaveToolkit(api_wrapper=api, include_gas_relay=True).get_tools()[3]
    assert isinstance(off, CleaveGetGas) and off.allow_relay is False
    assert isinstance(on, CleaveGetGas) and on.allow_relay is True
    assert on.invoke(SIGNED) == {"requestId": GAS_ID}
    assert len(seen) == 1


def test_api_key_is_sent_as_bearer() -> None:
    api, seen = mocked(answer(200, FIND_BODY), api_key="cq_live_test")
    CleaveFindToken(api_wrapper=api).invoke({"chainId": 1, "query": "USDC"})
    assert seen[0].headers["authorization"] == "Bearer cq_live_test"
    assert "cq_live_test" not in repr(api)


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEAVE_API_KEY", "cq_live_env")
    api = CleaveAPIWrapper()
    assert api.api_key is not None
    assert api.api_key.get_secret_value() == "cq_live_env"


def test_explicit_none_api_key_is_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEAVE_API_KEY", "cq_live_env")
    api, seen = mocked(answer(200, FIND_BODY), api_key=None)
    assert api.api_key is None
    CleaveFindToken(api_wrapper=api).invoke({"chainId": 1, "query": "USDC"})
    assert "authorization" not in seen[0].headers


def test_argument_key_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEAVE_API_KEY", "cq_live_env")
    api = CleaveAPIWrapper(api_key="cq_live_arg")
    assert api.api_key is not None
    assert api.api_key.get_secret_value() == "cq_live_arg"


def test_no_env_key_is_keyless() -> None:
    assert CleaveAPIWrapper().api_key is None


def test_default_timeout_is_sixty_seconds() -> None:
    assert CleaveAPIWrapper().timeout == 60.0


@pytest.mark.parametrize("timeout", [None, 7.5], ids=["default", "custom"])
def test_timeout_reaches_every_request(timeout: float | None) -> None:
    extra = {} if timeout is None else {"timeout": timeout}
    api, seen = mocked(answer(200, FIND_BODY), **extra)
    tool = CleaveFindToken(api_wrapper=api)
    tool.invoke({"chainId": 1, "query": "USDC"})
    asyncio.run(tool.ainvoke({"chainId": 1, "query": "USDC"}))
    expected = 60.0 if timeout is None else timeout
    for request in seen:
        assert request.extensions["timeout"] == {
            "connect": expected,
            "read": expected,
            "write": expected,
            "pool": expected,
        }
    assert len(seen) == 2


def test_non_json_429_is_retryable() -> None:
    api, _ = mocked(lambda request: httpx.Response(429, text="slow down"))
    tool = CleaveFindToken(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        tool.invoke({"chainId": 1, "query": "USDC"})
    assert caught.value.code == "http_error"
    assert caught.value.retryable is True


def test_non_json_4xx_is_not_retryable() -> None:
    api, _ = mocked(lambda request: httpx.Response(403, text="forbidden"))
    tool = CleaveFindToken(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        tool.invoke({"chainId": 1, "query": "USDC"})
    assert caught.value.retryable is False


def test_base_url_override() -> None:
    api, seen = mocked(answer(200, FIND_BODY), base_url="https://example.test/")
    CleaveFindToken(api_wrapper=api).invoke({"chainId": 1, "query": "USDC"})
    assert str(seen[0].url) == "https://example.test/agent/v1/find_token"


def test_toolkit_returns_the_four_tools_in_mcp_order() -> None:
    api, _ = mocked(answer(200, {}))
    tools = CleaveToolkit(api_wrapper=api).get_tools()
    assert [t.name for t in tools] == [
        "cleave_plan_swap",
        "cleave_track_swap",
        "cleave_find_token",
        "cleave_get_gas",
    ]
    assert all(getattr(t, "api_wrapper") is api for t in tools)


def test_default_toolkit_builds() -> None:
    tools = CleaveToolkit().get_tools()
    assert len(tools) == 4
    assert all(t.description for t in tools)


MCP_INPUTS = {
    "cleave_plan_swap": (
        {
            "fromChainId",
            "toChainId",
            "sellToken",
            "buyToken",
            "amount",
            "wallet",
            "recipient",
            "maxSlippageBps",
        },
        {"fromChainId", "sellToken", "buyToken", "amount", "wallet"},
    ),
    "cleave_track_swap": ({"chainId", "txHash", "requestId"}, set()),
    "cleave_find_token": ({"chainId", "query"}, {"chainId", "query"}),
    "cleave_get_gas": (
        {
            "payChainId",
            "chainId",
            "recipient",
            "amount",
            "wallet",
            "refundTo",
            "offer",
            "signature",
        },
        set(),
    ),
}


@pytest.mark.parametrize("tool", CleaveToolkit().get_tools(), ids=lambda t: t.name)
def test_schema_mirrors_mcp_inputs(tool: BaseTool) -> None:
    schema = convert_to_openai_tool(tool)["function"]["parameters"]
    fields, required = MCP_INPUTS[tool.name]
    assert set(schema["properties"]) == fields
    assert set(schema.get("required", [])) == required
    chain_fields = [k for k in fields if k.endswith("ChainId") or k == "chainId"]
    for key in chain_fields:
        prop = schema["properties"][key]
        options = prop.get("anyOf", [prop])
        enums = [o["enum"] for o in options if "enum" in o]
        assert enums == [[1, 10, 56, 137, 8453, 42161]]


VENDORED = json.loads(
    (Path(__file__).parent / "openapi_request_schemas.json").read_text()
)
INPUT_MODELS: dict[str, type[BaseModel]] = {
    "plan_swap": PlanSwapInput,
    "track_swap": TrackSwapInput,
    "find_token": FindTokenInput,
    "get_gas": GetGasInput,
}
PACKAGE_ONLY = {"description", "title"}


def _bare(prop: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in prop.items() if k not in PACKAGE_ONLY}


@pytest.mark.parametrize("tool", sorted(INPUT_MODELS))
def test_schema_snapshot_matches_vendored_openapi(tool: str) -> None:
    ref = VENDORED[tool]
    mine = INPUT_MODELS[tool].model_json_schema()
    assert set(mine) == {
        "additionalProperties",
        "properties",
        "required",
        "title",
        "type",
    } - (set() if "required" in ref else {"required"})
    assert mine["type"] == ref["type"] == "object"
    assert mine["additionalProperties"] is ref["additionalProperties"] is False
    assert mine.get("required", []) == ref.get("required", [])
    assert list(mine["properties"]) == list(ref["properties"])
    for field, prop in ref["properties"].items():
        assert _bare(mine["properties"][field]) == _bare(prop), field


@pytest.mark.parametrize("tool", CleaveToolkit().get_tools(), ids=lambda t: t.name)
def test_model_facing_schema_has_no_nulls_or_defaults(tool: BaseTool) -> None:
    params = convert_to_openai_tool(tool)["function"]["parameters"]
    assert params["additionalProperties"] is False
    ref = VENDORED[tool.name.removeprefix("cleave_")]
    for field, prop in params["properties"].items():
        assert "anyOf" not in prop
        assert "default" not in prop
        assert prop.get("type") != "null"
        assert _bare(prop) == _bare(ref["properties"][field]), field


BACKEND_TEXTS = json.loads(
    (Path(__file__).parent / "backend_tool_texts.json").read_text(encoding="utf-8")
)
REST_PLAN_EDITS = [
    (
        "sets requiresConfirmation, and the connection's settings can turn either "
        "into a refusal.",
        "sets requiresConfirmation.",
    ),
    (
        "Limits set in the connection (a pinned wallet, a per-plan USD cap, listed "
        "tokens only) bound every argument.",
        "The token to receive must be on Cleave's token list.",
    ),
]


def test_descriptions_are_backend_texts() -> None:
    assert _descriptions.TRACK_SWAP == BACKEND_TEXTS["track_swap"]
    assert _descriptions.FIND_TOKEN == BACKEND_TEXTS["find_token"]
    assert _descriptions.GET_GAS == BACKEND_TEXTS["get_gas"]
    plan = BACKEND_TEXTS["plan_swap"]
    for old, new in REST_PLAN_EDITS:
        assert plan.count(old) == 1
        plan = plan.replace(old, new)
    assert _descriptions.PLAN_SWAP == plan


def test_tool_descriptions_add_only_package_suffixes() -> None:
    assert CleavePlanSwap().description == _descriptions.PLAN_SWAP + PLAN_CONFIRM
    assert CleaveTrackSwap().description == _descriptions.TRACK_SWAP
    assert CleaveFindToken().description == _descriptions.FIND_TOKEN
    assert CleaveGetGas(allow_relay=True).description == (
        _descriptions.GET_GAS + GAS_SUBMIT
    )
    assert CleaveGetGas().description == _descriptions.GET_GAS + GAS_SUBMIT + RELAY_OFF
    assert "50 bps" in PLAN_CONFIRM
    assert "g:<payChainId>:<orderHash>" in GAS_SUBMIT


ORDER_HASH = "0x" + "ab" * 32
OFFER_BODY = {**GAS_BODY, "offer": "g2.known", "orderHash": ORDER_HASH}
PHASE1 = {"payChainId": 8453, "chainId": 8453, "recipient": WALLET, "amount": "5"}


def submitting(fail: Callable[[httpx.Request], httpx.Response]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if "offer" in json.loads(request.content):
            return fail(request)
        return httpx.Response(200, json=OFFER_BODY)

    return handler


def read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


FAILURES = [
    pytest.param(read_timeout, 0, id="timeout"),
    pytest.param(lambda r: httpx.Response(502, text="bad gateway"), 502, id="5xx"),
    pytest.param(lambda r: httpx.Response(200, json=["x"]), 200, id="bad-body"),
]


def unknown_error(content: Any) -> dict[str, Any]:
    error = json.loads(str(content))["error"]
    assert error["code"] == "outcome_unknown"
    assert error["retryable"] is False
    return error


@pytest.mark.parametrize(("fail", "status"), FAILURES)
def test_signed_submission_failure_is_outcome_unknown(
    fail: Callable[[httpx.Request], httpx.Response], status: int
) -> None:
    api, seen = mocked(submitting(fail))
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True)
    offer = tool.invoke(PHASE1)
    message = tool.invoke(
        call(tool, {"offer": offer["offer"], "signature": "0x1b", "wallet": WALLET})
    )
    assert message.status == "error"
    error = unknown_error(message.content)
    assert error["payChainId"] == 8453
    assert error["orderHash"] == ORDER_HASH
    assert error["trackRequestId"] == f"g:8453:{ORDER_HASH}"
    assert f"g:8453:{ORDER_HASH}" in error["message"]
    assert len(seen) == 2


@pytest.mark.parametrize(("fail", "status"), FAILURES)
async def test_signed_submission_failure_is_outcome_unknown_async(
    fail: Callable[[httpx.Request], httpx.Response], status: int
) -> None:
    api, _ = mocked(submitting(fail))
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True, handle_tool_error=False)
    offer = await tool.ainvoke(PHASE1)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(
            {"offer": offer["offer"], "signature": "0x1b", "wallet": WALLET}
        )
    assert caught.value.code == "outcome_unknown"
    assert caught.value.retryable is False
    assert caught.value.body["error"]["orderHash"] == ORDER_HASH


def test_unknown_offer_outcome_has_no_order_reference() -> None:
    api, _ = mocked(submitting(read_timeout))
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True)
    error = unknown_error(
        tool.invoke({"offer": "g2.never.seen", "signature": "0x1b", "wallet": WALLET})
    )
    assert "orderHash" not in error
    assert "g:<payChainId>:<orderHash>" in error["message"]


def test_definite_refusal_on_submission_passes_through() -> None:
    used = {
        "apiVersion": "1",
        "requestId": "r",
        "error": {
            "code": "invalid_param",
            "reason": "authorization_used",
            "message": "m",
            "retryable": False,
        },
    }
    api, _ = mocked(submitting(lambda r: httpx.Response(400, json=used)))
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True)
    offer = tool.invoke(PHASE1)
    content = tool.invoke(
        {"offer": offer["offer"], "signature": "0x1b", "wallet": WALLET}
    )
    assert json.loads(content) == used


def test_offer_request_failure_stays_retryable_network_error() -> None:
    api, _ = mocked(read_timeout)
    error = json.loads(CleaveGetGas(api_wrapper=api).invoke(PHASE1))["error"]
    assert error["code"] == "network_error"
    assert error["retryable"] is True


def test_remembered_offers_are_bounded() -> None:
    counter = iter(range(1000))

    def handler(request: httpx.Request) -> httpx.Response:
        n = next(counter)
        return httpx.Response(
            200, json={**OFFER_BODY, "offer": f"g2.{n}", "orderHash": ORDER_HASH}
        )

    api, _ = mocked(handler)
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True)
    for _ in range(OFFERS_KEPT + 5):
        tool.invoke(PHASE1)
    assert len(tool._offers) == OFFERS_KEPT
    assert "g2.0" not in tool._offers


def inheriting_clients(handler: Handler) -> dict[str, Any]:
    transport = httpx.MockTransport(handler)
    inherited = {"Authorization": "Bearer inherited"}
    return {
        "client": httpx.Client(
            transport=transport, headers=inherited, auth=httpx.BasicAuth("u", "p")
        ),
        "async_client": httpx.AsyncClient(
            transport=transport, headers=inherited, auth=httpx.BasicAuth("u", "p")
        ),
    }


@pytest.mark.parametrize(
    ("key", "expected"),
    [(None, None), ("cq_live_mine", "Bearer cq_live_mine")],
    ids=["keyless", "keyed"],
)
def test_client_credentials_never_leak(
    monkeypatch: pytest.MonkeyPatch, key: str | None, expected: str | None
) -> None:
    monkeypatch.setenv("CLEAVE_API_KEY", "cq_live_env")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=FIND_BODY)

    api = CleaveAPIWrapper(api_key=key, **inheriting_clients(handler))
    tool = CleaveFindToken(api_wrapper=api)
    tool.invoke({"chainId": 1, "query": "USDC"})
    asyncio.run(tool.ainvoke({"chainId": 1, "query": "USDC"}))
    assert len(seen) == 2
    assert [r.headers.get("authorization") for r in seen] == [expected, expected]
