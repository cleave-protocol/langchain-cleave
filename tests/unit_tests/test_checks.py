from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from Crypto.Hash import keccak
from langchain_core.messages import ToolMessage

from langchain_cleave import (
    CheckResult,
    CleaveAPIError,
    CleaveAPIWrapper,
    CleaveGetGas,
    CleavePlanSwap,
    PlanSwapInput,
    check_get_gas,
    check_plan_swap,
)
from langchain_cleave._allowlist import ALLOWLIST
from langchain_cleave._checks import screen_get_gas, screen_plan_swap

FIXTURES = Path(__file__).parent / "fixtures"
FILE = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
VECTORS: list[dict[str, Any]] = FILE["vectors"]
RULES = ["G1", "G2", "G3", "G4", "P1", "P2", "P3", "P4", "P5", "P6"]


def by_name(name: str) -> dict[str, Any]:
    return copy.deepcopy(next(v for v in VECTORS if v["name"] == name))


def run_check(v: dict[str, Any]) -> CheckResult:
    if v["tool"] == "plan_swap":
        return check_plan_swap(v["request"], v["response"])
    return check_get_gas(v["request"], v["response"], now_of(v))


def now_of(v: dict[str, Any]) -> int | None:
    return v.get("context", {}).get("nowSec")


def screen(v: dict[str, Any]) -> dict[str, Any]:
    if v["tool"] == "plan_swap":
        return screen_plan_swap(v["request"], v["response"])
    return screen_get_gas(v["request"], v["response"], now_of(v))


def verdict(v: dict[str, Any]) -> dict[str, Any]:
    out = screen(v)
    error = out.get("error")
    if isinstance(error, dict) and error.get("code") == "address_not_allowed":
        assert isinstance(error["message"], str)
        assert "malformed" not in error["message"]
        assert list(out) == ["error"]
        return {"result": "fail", "rule": error["message"][:2]}
    if v["expect"]["result"] == "pass" and "output" in v["expect"]:
        return {"result": "pass", "output": out}
    return {"result": "pass"}


def test_vectors_cover_every_rule_live_probes_and_the_table_version() -> None:
    rules = {v["expect"]["rule"] for v in VECTORS if v["expect"]["result"] == "fail"}
    assert sorted(rules) == RULES
    assert len([v for v in VECTORS if v["name"].startswith("live ")]) >= 30
    assert len(VECTORS) == 259
    assert FILE["allowlistVersion"] == ALLOWLIST["version"]


@pytest.mark.parametrize("v", VECTORS, ids=[v["name"] for v in VECTORS])
def test_shared_vector(v: dict[str, Any]) -> None:
    assert verdict(v) == v["expect"]


NEGATIVES = [
    ("call for another chain", "P1: call 1 is for another chain"),
    ("attacker main to", "P2: the main call's target is not on the allowlist"),
    ("attacker spender", "P3: call 0 approves another spender"),
    ("main call spends another token", "P4: the main call spends another token"),
    ("extra value", "P5: the main call sends another native value"),
    (
        "same-chain attacker referrer at the 100 bps cap",
        "P5: the referral fee goes to another address or has another size",
    ),
    ("recipient swapped in a swap", "P6: the swap pays another recipient"),
    ("wrong verifyingContract", "G1: the authorization is for another token"),
    ("gas recipient swapped", "G2: the order pays another recipient"),
    ("gas router swapped", "G3: order.router is not the pinned one"),
    ("gas message.to swapped", "G4: the authorization pays another address"),
]


@pytest.mark.parametrize(
    ("name", "message"), NEGATIVES, ids=[m[:2] for _, m in NEGATIVES]
)
def test_one_negative_per_rule_names_its_reason(name: str, message: str) -> None:
    r = run_check(by_name(name))
    assert (r.ok, r.rule, r.message) == (False, message[:2], message)


def pinned_table() -> Any:
    return json.loads((FIXTURES / "allowlist-pinned.json").read_text(encoding="utf-8"))


def keccak_hex(data: bytes) -> str:
    return "0x" + keccak.new(digest_bits=256, data=data).hexdigest()


def test_pinned_table_equals_the_shared_table() -> None:
    assert pinned_table() == ALLOWLIST


def test_init_code_hashes_recompute_from_the_templates() -> None:
    for kind in ("swap", "cross_chain"):
        spec = ALLOWLIST["gas"][kind]
        code = (
            "3d602d80600a3d3981f3363d3d373d3d3d363d73"
            + spec["template"][2:].lower()
            + "5af43d82803e903d91602b57fd5bf3"
        )
        assert keccak_hex(bytes.fromhex(code)) == spec["initCodeHash"]


P = "(address,address,uint256,uint256,uint256,address,address,uint16,bool)"
CROSS = "(uint256,uint256),bytes,bytes32,uint256,address,address,uint256,bytes,bytes"
SIGNATURES = {
    "swapWithBlob": f"swapWithBlob({P},bytes)",
    "swapMeasuredWithBlobV1": f"swapMeasuredWithBlobV1({P},(uint256,uint256),bytes)",
    "crossChainSwap": f"crossChainSwap({P},{CROSS})",
    "crossChainSwapNative": f"crossChainSwapNative({P},{CROSS})",
    "bridgeToken": (
        "bridgeToken(address,uint256,bytes32,uint256,address,address,uint256,"
        "address,uint16,bytes,bytes)"
    ),
    "deposit": "deposit()",
    "withdraw": "withdraw(uint256)",
    "depositNative": (
        "depositNative(address,address,bytes32,address,uint256,bytes32,uint256,"
        "uint256,bytes32,uint32,uint32,uint32,bytes)"
    ),
}
SPOKE_DEPOSIT = (
    "deposit(bytes32,bytes32,bytes32,bytes32,uint256,uint256,uint256,bytes32,"
    "uint32,uint32,uint32,bytes)"
)


def test_each_selector_is_named_after_its_function_signature() -> None:
    for chain in ALLOWLIST["chains"].values():
        for target in chain["targets"].values():
            for selector, name in target["selectors"].items():
                if target["role"] == "across_spoke_pool":
                    signature = SPOKE_DEPOSIT
                else:
                    signature = SIGNATURES[name]
                assert keccak_hex(signature.encode())[:10] == selector, name


ACROSS_RAILS = (
    "0xf6c86ecb0564cd8d732c9d68c369fbc03a0c351db7419e70be166d26a2ed4916",
    "0xa28bababc0e2b0c1c28d78d20a75d5ef0afd16dc155828d76a2bf6b97805d328",
)


def test_every_lane_has_its_rails_with_the_chain_id_as_across_destination() -> None:
    chains = ALLOWLIST["chains"]
    for source, chain in chains.items():
        lanes = chain["lanes"]
        assert sorted(lanes) == sorted(k for k in chains if k != source)
        for to, rails in lanes.items():
            for rail in ACROSS_RAILS:
                assert rails[rail] == int(to), f"{source}>{to}"
            if chain["gasPay"] is not None:
                assert rails[chain["gasPay"]["railId"]] == int(to)


def test_six_targets_on_every_chain_and_gas_payment_on_five() -> None:
    chains = ALLOWLIST["chains"]
    assert sorted(chains) == ["1", "10", "137", "42161", "56", "8453"]
    assert all(len(c["targets"]) == 6 for c in chains.values())
    paying = sorted(k for k, c in chains.items() if c["gasPay"] is not None)
    assert paying == ["1", "10", "137", "42161", "8453"]


def answering(body: Any) -> tuple[CleaveAPIWrapper, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    api = CleaveAPIWrapper(
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        api_key=None,
    )
    return api, seen


def refusal(message: str) -> dict[str, Any]:
    return {
        "error": {"code": "address_not_allowed", "message": message, "retryable": False}
    }


def test_plan_swap_returns_a_checked_plan_unchanged() -> None:
    v = by_name("live plan-8453-usdc-x42161-weth")
    api, seen = answering(v["response"])
    assert CleavePlanSwap(api_wrapper=api).invoke(v["request"]) == v["response"]
    assert len(seen) == 1


async def test_plan_swap_returns_a_checked_plan_unchanged_async() -> None:
    v = by_name("live plan-8453-usdc-x42161-weth")
    api, _ = answering(v["response"])
    assert await CleavePlanSwap(api_wrapper=api).ainvoke(v["request"]) == v["response"]


def test_plan_swap_keeps_only_the_documented_plan_fields() -> None:
    v = by_name("extra call fields and top-level keys are dropped")
    api, _ = answering(v["response"])
    assert CleavePlanSwap(api_wrapper=api).invoke(v["request"]) == v["expect"]["output"]


async def test_get_gas_keeps_only_the_documented_offer_fields_async() -> None:
    v = by_name("extra offer keys and typed data fields are dropped")
    api, _ = answering(v["response"])
    tool = CleaveGetGas(api_wrapper=api)
    assert await tool.ainvoke(v["request"]) == v["expect"]["output"]


def test_get_gas_submission_keeps_only_its_four_fields() -> None:
    v = by_name("phase 2 answer keeps its four fields")
    api, _ = answering(v["response"])
    tool = CleaveGetGas(api_wrapper=api, allow_relay=True)
    assert tool.invoke(v["request"]) == v["expect"]["output"]


def test_plan_swap_replaces_a_failing_plan_with_the_refusal_only() -> None:
    v = by_name("attacker spender")
    api, _ = answering(v["response"])
    content = CleavePlanSwap(api_wrapper=api).invoke(v["request"])
    assert json.loads(content) == refusal("P3: call 0 approves another spender")


async def test_plan_swap_replaces_a_failing_plan_with_the_refusal_only_async() -> None:
    v = by_name("attacker spender")
    api, _ = answering(v["response"])
    content = await CleavePlanSwap(api_wrapper=api).ainvoke(v["request"])
    assert json.loads(content) == refusal("P3: call 0 approves another spender")


def test_a_failing_plan_is_an_error_tool_message() -> None:
    v = by_name("attacker main to")
    api, _ = answering(v["response"])
    tool = CleavePlanSwap(api_wrapper=api)
    message = tool.invoke(
        {"name": tool.name, "args": v["request"], "id": "call-1", "type": "tool_call"}
    )
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert json.loads(str(message.content)) == refusal(
        "P2: the main call's target is not on the allowlist"
    )


async def test_a_failing_plan_raises_when_unhandled_async() -> None:
    v = by_name("extra value")
    api, _ = answering(v["response"])
    tool = CleavePlanSwap(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(v["request"])
    assert caught.value.code == "address_not_allowed"
    assert caught.value.retryable is False
    assert caught.value.status is None
    assert caught.value.body == refusal("P5: the main call sends another native value")


def test_get_gas_replaces_a_failing_offer_with_the_refusal_only() -> None:
    v = by_name("gas message.to swapped")
    api, _ = answering(v["response"])
    tool = CleaveGetGas(api_wrapper=api)
    content = tool.invoke(v["request"])
    assert json.loads(content) == refusal("G4: the authorization pays another address")
    assert len(tool._offers) == 0


async def test_get_gas_replaces_a_failing_offer_with_the_refusal_only_async() -> None:
    v = by_name("gas recipient swapped")
    api, _ = answering(v["response"])
    tool = CleaveGetGas(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(v["request"])
    assert caught.value.body == refusal("G2: the order pays another recipient")
    assert len(tool._offers) == 0


def test_get_gas_returns_a_checked_offer_unchanged() -> None:
    v = by_name("live gas-8453-x42161-recipient")
    api, _ = answering(v["response"])
    tool = CleaveGetGas(api_wrapper=api)
    assert tool.invoke(v["request"]) == v["response"]
    assert list(tool._offers) == [v["response"]["offer"]]


async def test_get_gas_returns_a_checked_offer_unchanged_async() -> None:
    v = by_name("live gas-1-same")
    api, _ = answering(v["response"])
    assert await CleaveGetGas(api_wrapper=api).ainvoke(v["request"]) == v["response"]


def test_a_body_whose_error_member_is_not_an_object_is_checked() -> None:
    v = by_name("error member null with an attacker target")
    api, _ = answering(v["response"])
    content = CleavePlanSwap(api_wrapper=api).invoke(v["request"])
    assert json.loads(content) == refusal(
        "P2: the main call's target is not on the allowlist"
    )


API_REFUSAL = {
    "apiVersion": "1",
    "requestId": "r-9",
    "error": {
        "code": "invalid_param",
        "reason": "bad_amount",
        "message": "x",
        "retryable": False,
        "extra": "dropped",
    },
    "calls": [{"to": "0x000000000000000000000000000000000000dEaD"}],
    "typedData": {"message": {"to": "0x000000000000000000000000000000000000dEaD"}},
}
KEPT_REFUSAL = {
    "apiVersion": "1",
    "requestId": "r-9",
    "error": {
        "code": "invalid_param",
        "reason": "bad_amount",
        "message": "x",
        "retryable": False,
    },
}


def tool_args(cls: type[CleavePlanSwap] | type[CleaveGetGas]) -> dict[str, Any]:
    name = "live plan-8453-same" if cls is CleavePlanSwap else "live gas-1-same"
    return by_name(name)["request"]


@pytest.mark.parametrize("cls", [CleavePlanSwap, CleaveGetGas])
async def test_an_api_refusal_keeps_only_its_error_fields(
    cls: type[CleavePlanSwap] | type[CleaveGetGas],
) -> None:
    api, _ = answering(API_REFUSAL)
    assert json.loads(cls(api_wrapper=api).invoke(tool_args(cls))) == KEPT_REFUSAL
    tool = cls(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(tool_args(cls))
    assert caught.value.status == 200
    assert caught.value.body == KEPT_REFUSAL


@pytest.mark.parametrize("cls", [CleavePlanSwap, CleaveGetGas])
async def test_an_http_api_refusal_keeps_only_its_error_fields(
    cls: type[CleavePlanSwap] | type[CleaveGetGas],
) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(400, json=API_REFUSAL))
    api = CleaveAPIWrapper(
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        api_key=None,
    )
    assert json.loads(cls(api_wrapper=api).invoke(tool_args(cls))) == KEPT_REFUSAL
    tool = cls(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(CleaveAPIError) as caught:
        await tool.ainvoke(tool_args(cls))
    assert caught.value.status == 400
    assert caught.value.body == KEPT_REFUSAL


PLAN = by_name("live plan-8453-same")


def test_a_malformed_body_fails_closed() -> None:
    gas = by_name("live gas-1-same")["request"]
    bad = check_plan_swap(PLAN["request"], {"calls": "nope"})
    assert (bad.ok, bad.rule) == (False, "P1")
    empty = check_plan_swap(PLAN["request"], {"blocked": False, "calls": []})
    assert (empty.ok, empty.rule) == (False, "P1")
    assert check_get_gas(gas, {}).rule == "G1"
    walletless = {k: v for k, v in gas.items() if k != "wallet"}
    assert check_get_gas(walletless, {}).rule == "G2"


def test_a_blocked_plan_without_calls_passes() -> None:
    body = {"blocked": True, "calls": []}
    assert check_plan_swap(PLAN["request"], body) == CheckResult(True)


def test_addresses_compare_without_regard_to_case() -> None:
    def upper(a: str) -> str:
        return "0x" + a[2:].upper()

    body = copy.deepcopy(PLAN["response"])
    body["send"]["token"] = upper(body["send"]["token"])
    body["calls"] = [
        {**c, "from": upper(c["from"]), "to": upper(c["to"]), "data": upper(c["data"])}
        for c in body["calls"]
    ]
    request = {**PLAN["request"], "wallet": PLAN["request"]["wallet"].lower()}
    assert check_plan_swap(request, body) == CheckResult(True)


def test_the_message_names_the_failing_rule() -> None:
    r = check_plan_swap({**PLAN["request"], "fromChainId": 1}, PLAN["response"])
    assert r == CheckResult(False, "P1", "P1: call 0 is for another chain")


def test_a_boolean_chain_id_is_not_chain_1() -> None:
    v = by_name("quote-derived quote-1-usdc-weth")
    assert check_plan_swap(v["request"], v["response"]) == CheckResult(True)
    send = copy.deepcopy(v["response"])
    send["send"]["chainId"] = True
    r = check_plan_swap(v["request"], send)
    assert r == CheckResult(False, "P3", "P3: send is on another chain")
    call = copy.deepcopy(v["response"])
    call["calls"][0]["chainId"] = True
    r = check_plan_swap(v["request"], call)
    assert r == CheckResult(False, "P1", "P1: call 0 is for another chain")
    r = check_plan_swap({**v["request"], "fromChainId": True}, v["response"])
    assert r.rule == "P1"


def test_a_request_model_and_omitted_none_fields_are_accepted() -> None:
    model = PlanSwapInput(**PLAN["request"])
    assert check_plan_swap(model, PLAN["response"]) == CheckResult(True)
    request = {**PLAN["request"], "recipient": None, "toChainId": None}
    assert check_plan_swap(request, PLAN["response"]) == CheckResult(True)


def test_a_boolean_is_not_an_unsigned_integer() -> None:
    v = by_name("live gas-1-same")
    assert check_get_gas(v["request"], v["response"]) == CheckResult(True)
    v["response"]["typedData"]["message"]["validAfter"] = False
    r = check_get_gas(v["request"], v["response"])
    assert r == CheckResult(False, "G1", "G1: validAfter is not an unsigned integer")


def test_an_address_with_a_trailing_newline_is_not_an_address() -> None:
    body = copy.deepcopy(PLAN["response"])
    body["send"]["token"] += "\n"
    r = check_plan_swap(PLAN["request"], body)
    assert r == CheckResult(False, "P3", "P3: send.token is not an address")


def test_a_stablecoin_sell_to_an_unpinned_token_may_carry_no_referrer() -> None:
    v = by_name("live plan-8453-usdc-x42161-recipient")
    v["response"]["receive"].pop("token", None)
    request = {**v["request"], "buyToken": "SOMECOIN"}
    assert check_plan_swap(request, v["response"]) == CheckResult(True)


def test_the_gas_deadline_bound_uses_the_clock_by_default() -> None:
    v = by_name("live gas-1-same")
    v["response"]["order"]["deadline"] = int(time.time()) + 5000
    late = "G2: the order deadline is too far away"
    assert check_get_gas(v["request"], v["response"]).message == late
    deadline = v["response"]["order"]["deadline"]
    r = check_get_gas(v["request"], v["response"], deadline)
    assert r.rule == "G4"


def test_a_padded_buy_symbol_is_pinned_after_trimming() -> None:
    v = by_name("stablecoin bridge charged the house fee")
    del v["response"]["receive"]["token"]
    request = {**v["request"], "buyToken": " USDC "}
    r = check_plan_swap(request, v["response"])
    assert (
        r.message == "P5: the referral fee goes to another address or has another size"
    )
