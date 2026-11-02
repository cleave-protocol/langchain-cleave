from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from Crypto.Hash import keccak
from pydantic import BaseModel

from langchain_cleave._allowlist import ALLOWLIST
from langchain_cleave._api import refusal_body

CheckRule = Literal["P1", "P2", "P3", "P4", "P5", "P6", "G1", "G2", "G3", "G4", "G5"]


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    rule: CheckRule | None = None
    message: str | None = None


@dataclass(frozen=True)
class _GasPay:
    router_v3: str
    across_usdc_adapter: str


@dataclass(frozen=True)
class _Chain:
    usdc: str
    usdc_decimals: int
    wrapped_native: str
    composer: str
    delivery_shell: str
    gas_pay: _GasPay | None
    targets: dict[str, frozenset[str]]
    spoke_pool: str | None


@dataclass(frozen=True)
class _GasKind:
    template: str
    version_tag: str
    init_code_hash: str


@dataclass(frozen=True)
class _Call:
    to: str
    data: str
    value: int


@dataclass(frozen=True)
class _Pulls:
    token: int
    pulled: tuple[int, ...]
    amount_in: int


def _chain(entry: dict[str, Any]) -> _Chain:
    targets: dict[str, Any] = entry["targets"]
    pay = entry["gasPay"]
    return _Chain(
        usdc=entry["usdc"].lower(),
        usdc_decimals=entry["usdcDecimals"],
        wrapped_native=entry["wrappedNative"].lower(),
        composer=entry["composer"].lower(),
        delivery_shell=entry["gasDeliveryShell"].lower(),
        gas_pay=None
        if pay is None
        else _GasPay(pay["routerV3"].lower(), pay["acrossUsdcAdapter"].lower()),
        targets={a.lower(): frozenset(t["selectors"]) for a, t in targets.items()},
        spoke_pool=next(
            (a.lower() for a, t in targets.items() if t["role"] == "across_spoke_pool"),
            None,
        ),
    )


def _gas_kind(entry: dict[str, Any]) -> _GasKind:
    return _GasKind(
        entry["template"].lower(),
        entry["versionTag"].lower(),
        entry["initCodeHash"].lower(),
    )


CHAINS = {int(k): _chain(v) for k, v in ALLOWLIST["chains"].items()}
NATIVE = ALLOWLIST["nativeSentinel"].lower()
HANDLER = ALLOWLIST["acrossHandler"].lower()
FACTORY = ALLOWLIST["gas"]["factory"].lower()
GAS_SWAP = _gas_kind(ALLOWLIST["gas"]["swap"])
GAS_CROSS_CHAIN = _gas_kind(ALLOWLIST["gas"]["cross_chain"])
MAX_UINT = (1 << 256) - 1
MAX_SAFE_INTEGER = (1 << 53) - 1
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
HEX_DATA = re.compile(r"0x(?:[0-9a-fA-F]{2})*")
HEX_QUANTITY = re.compile(r"0x[0-9a-fA-F]{1,64}")
BYTES32 = re.compile(r"0x[0-9a-fA-F]{64}")
DECIMAL_INT = re.compile(r"[0-9]{1,78}")
DECIMAL = re.compile(r"([0-9]{1,30})(?:\.([0-9]{1,30}))?")

APPROVE = "0x095ea7b3"
SWAP = "0xab33b1ec"
MEASURED = "0xe0b7822f"
CROSS = "0x84022178"
CROSS_NATIVE = "0xf79eec2e"
BRIDGE = "0x9edd28ad"
DEPOSIT = "0xd0e30db0"
WITHDRAW = "0x2e1a7d4d"
SPOKE_DEPOSIT = "0xad5425c6"
DEPOSIT_NATIVE = "0x609ea081"

PULLS = {
    SWAP: _Pulls(0, (2,), 2),
    MEASURED: _Pulls(0, (9,), 2),
    CROSS: _Pulls(0, (2, 9), 2),
    BRIDGE: _Pulls(0, (1,), 1),
    SPOKE_DEPOSIT: _Pulls(2, (4,), 4),
}

FieldType = Literal["uint", "uint8", "address", "bytes32"]

SWAP_ORDER: tuple[tuple[str, FieldType], ...] = (
    ("chainId", "uint"),
    ("token", "address"),
    ("shell", "address"),
    ("amount", "uint"),
    ("fee", "uint"),
    ("feeRecipient", "address"),
    ("maxProtocolFeeBps", "uint"),
    ("refundTo", "address"),
    ("recipient", "address"),
    ("minOut", "uint"),
    ("deadline", "uint"),
    ("quoteId", "bytes32"),
)

ACROSS_ORDER: tuple[tuple[str, FieldType], ...] = (
    ("chainId", "uint"),
    ("token", "address"),
    ("router", "address"),
    ("railId", "bytes32"),
    ("adapter", "address"),
    ("dstChainId", "uint"),
    ("dstToken", "address"),
    ("dstDecimals", "uint8"),
    ("handler", "address"),
    ("dstShell", "address"),
    ("dstWrappedNative", "address"),
    ("amount", "uint"),
    ("fee", "uint"),
    ("feeRecipient", "address"),
    ("maxProtocolFeeBps", "uint"),
    ("relayFee", "uint"),
    ("refundTo", "address"),
    ("recipient", "address"),
    ("outputAmount", "uint"),
    ("minOut", "uint"),
    ("messageHash", "bytes32"),
    ("deadline", "uint"),
    ("quoteId", "bytes32"),
)

AUTH_FIELDS = (
    ("from", "address"),
    ("to", "address"),
    ("value", "uint256"),
    ("validAfter", "uint256"),
    ("validBefore", "uint256"),
    ("nonce", "bytes32"),
)


class _Refusal(Exception):
    def __init__(self, rule: CheckRule, message: str) -> None:
        super().__init__(message)
        self.rule: CheckRule = rule
        self.message = message


class _Step:
    def __init__(self) -> None:
        self.rule: CheckRule = "P1"

    def __call__(self, rule: CheckRule) -> None:
        self.rule = rule


def _fail(rule: CheckRule, message: str) -> NoReturn:
    raise _Refusal(rule, message)


def _guard(run: Callable[[_Step], None]) -> CheckResult:
    step = _Step()
    try:
        run(step)
    except _Refusal as refusal:
        return CheckResult(False, refusal.rule, f"{refusal.rule}: {refusal.message}")
    except Exception:
        return CheckResult(False, step.rule, f"{step.rule}: the answer is malformed")
    return CheckResult(True)


def _request(request: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    fields = request.model_dump() if isinstance(request, BaseModel) else request
    return {k: v for k, v in fields.items() if v is not None}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _is_number(a) and _is_number(b):
        return bool(a == b)
    return type(a) is type(b) and bool(a == b)


def _lookup(chain_id: Any) -> _Chain | None:
    return CHAINS.get(chain_id) if _is_number(chain_id) else None


def _object(v: Any, rule: CheckRule, what: str) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    _fail(rule, f"{what} is missing")


def _address(v: Any, rule: CheckRule, what: str) -> str:
    if isinstance(v, str) and ADDRESS.fullmatch(v):
        return v.lower()
    _fail(rule, f"{what} is not an address")


def _integer(v: Any, rule: CheckRule, what: str) -> int:
    if (
        _is_number(v)
        and (isinstance(v, int) or v.is_integer())
        and 0 <= v <= MAX_SAFE_INTEGER
    ):
        return int(v)
    if isinstance(v, str) and DECIMAL_INT.fullmatch(v):
        n = int(v)
        if n <= MAX_UINT:
            return n
    _fail(rule, f"{what} is not an unsigned integer")


def _quantity(v: Any, rule: CheckRule, what: str) -> int:
    if isinstance(v, str) and HEX_QUANTITY.fullmatch(v):
        return int(v, 16)
    _fail(rule, f"{what} is not a hex quantity")


def _bytes32(v: Any, rule: CheckRule, what: str) -> str:
    if isinstance(v, str) and BYTES32.fullmatch(v):
        return v.lower()
    _fail(rule, f"{what} is not 32 bytes")


def _units(v: Any, decimals: int, rule: CheckRule, what: str) -> int:
    m = DECIMAL.fullmatch(v) if isinstance(v, str) else None
    frac = (m.group(2) or "") if m is not None else ""
    if m is None or len(frac) > decimals:
        _fail(rule, f"{what} is not a decimal amount")
    return int(m.group(1)) * 10**decimals + int(frac.ljust(decimals, "0") or "0")


def _word(data: str, k: int, rule: CheckRule, what: str) -> str:
    w = data[10 + 64 * k : 74 + 64 * k]
    if len(w) == 64:
        return w
    _fail(rule, f"{what} is missing from the calldata")


def _word_int(data: str, k: int, rule: CheckRule, what: str) -> int:
    return int(_word(data, k, rule, what), 16)


def _word_address(data: str, k: int, rule: CheckRule, what: str) -> str:
    w = _word(data, k, rule, what)
    if w.startswith("0" * 24):
        return "0x" + w[24:]
    _fail(rule, f"{what} is not an address")


def _same_chain(v: Any, chain_id: Any) -> bool:
    if _is_number(v):
        return _same(v, chain_id)
    if isinstance(v, str) and DECIMAL_INT.fullmatch(v):
        return int(v) == int(chain_id)
    return False


def _keccak(hex_text: str) -> str:
    data = bytes.fromhex(hex_text[2:])
    return "0x" + keccak.new(digest_bits=256, data=data).hexdigest()


def _call(raw: Any, i: int, from_chain: Any, wallet: str) -> _Call:
    c = _object(raw, "P1", f"call {i}")
    if not _same_chain(c.get("chainId"), from_chain):
        _fail("P1", f"call {i} is for another chain")
    if _address(c.get("from"), "P1", f"call {i} from") != wallet:
        _fail("P1", f"call {i} is from another wallet")
    to = _address(c.get("to"), "P1", f"call {i} to")
    data = c.get("data")
    if not (isinstance(data, str) and HEX_DATA.fullmatch(data) and len(data) >= 10):
        _fail("P1", f"call {i} data is malformed")
    return _Call(to, data.lower(), _quantity(c.get("value"), "P1", f"call {i} value"))


def _plan(req: dict[str, Any], body: Mapping[str, Any], at: _Step) -> None:
    at("P1")
    from_chain = req.get("fromChainId")
    to_chain = req.get("toChainId", from_chain)
    wallet = _address(req.get("wallet"), "P1", "wallet")
    calls = body.get("calls")
    if not isinstance(calls, list):
        _fail("P1", "calls are missing")
    if body.get("blocked") is True:
        if len(calls) > 0:
            _fail("P1", "a blocked plan carries calls")
        return
    if len(calls) == 0:
        _fail("P1", "the plan has no calls")
    parsed = [_call(raw, i, from_chain, wallet) for i, raw in enumerate(calls)]
    main = parsed[-1]
    selector = main.data[:10]

    at("P2")
    chain = _lookup(from_chain)
    if chain is None:
        _fail("P2", "the chain is not on the allowlist")
    allowed = chain.targets.get(main.to)
    if allowed is None:
        _fail("P2", "the main call's target is not on the allowlist")
    if selector not in allowed:
        _fail("P2", "the main call's function is not allowed on its target")

    at("P3")
    send = _object(body.get("send"), "P3", "send")
    if not _same_chain(send.get("chainId"), from_chain):
        _fail("P3", "send is on another chain")
    token = _address(send.get("token"), "P3", "send.token")
    sell = req.get("sellToken")
    if isinstance(sell, str) and ADDRESS.fullmatch(sell) and sell.lower() != token:
        _fail("P3", "send.token is not the sell token")
    native = token == NATIVE
    approves = parsed[:-1]
    if len(approves) > 2:
        _fail("P3", "more than two calls come before the main call")
    approved: list[int] = []
    for i, a in enumerate(approves):
        if not a.data.startswith(APPROVE) or len(a.data) != 138 or a.value != 0:
            _fail("P3", f"call {i} is not a plain approve")
        if native or a.to != token:
            _fail("P3", f"call {i} approves another token")
        if _word_address(a.data, 0, "P3", "spender") != main.to:
            _fail("P3", f"call {i} approves another spender")
        amount = _word_int(a.data, 1, "P3", "approve amount")
        if amount == MAX_UINT:
            _fail("P3", f"call {i} is an unlimited approve")
        approved.append(amount)
    last = approved[-1] if approved else None
    if last == 0:
        _fail("P3", "the last approve is zero")
    if len(approved) == 2 and approved[0] != 0:
        _fail("P3", "only approve(0) may come before the approve")

    at("P4")
    pulls = PULLS.get(selector)
    if pulls is None:
        if approves:
            _fail("P4", "this function takes no approve")
        if main.to != token if selector == WITHDRAW else not native:
            _fail("P4", "the main call spends another token")
    elif not native:
        if _word_address(main.data, pulls.token, "P4", "token") != token:
            _fail("P4", "the main call spends another token")
        if last is not None:
            pulled = max(_word_int(main.data, k, "P4", "amount") for k in pulls.pulled)
            if pulled != last:
                _fail("P4", "the main call pulls another amount than approved")
            if pulled > 2 * _word_int(main.data, pulls.amount_in, "P4", "amountIn"):
                _fail("P4", "the main call pulls more than twice its input")

    at("P5")
    fees = body.get("fees", [])
    if not isinstance(fees, list):
        _fail("P5", "fees are malformed")
    value = _units(send.get("amount"), 18, "P5", "send.amount") if native else 0
    for f in fees:
        fee = _object(f, "P5", "fee")
        if fee.get("kind") == "native_messaging":
            value += _units(fee.get("amount"), 18, "P5", "fee amount")
    if main.value != value:
        _fail("P5", "the main call sends another native value")

    at("P6")
    recipient = (
        _address(req["recipient"], "P6", "recipient") if "recipient" in req else wallet
    )
    destination = None if _same(to_chain, from_chain) else _lookup(to_chain)
    composer = None if destination is None else destination.composer

    def arrives(a: str) -> bool:
        return a in (recipient, HANDLER, composer)

    def at6(k: int, what: str) -> str:
        return _word_address(main.data, k, "P6", what)

    if selector in (SWAP, MEASURED):
        if at6(5, "recipient") != recipient:
            _fail("P6", "the swap pays another recipient")
    elif selector in (CROSS, CROSS_NATIVE):
        if at6(5, "source recipient") != main.to:
            _fail("P6", "the source swap pays out of the router")
        if not arrives(at6(14, "recipient")):
            _fail("P6", "the transfer pays another recipient")
        if at6(15, "refund recipient") != wallet:
            _fail("P6", "refunds go to another address")
    elif selector == BRIDGE:
        if not arrives(at6(4, "recipient")):
            _fail("P6", "the transfer pays another recipient")
        if at6(5, "refund recipient") != wallet:
            _fail("P6", "refunds go to another address")
    elif selector == SPOKE_DEPOSIT:
        if at6(0, "depositor") != wallet:
            _fail("P6", "refunds go to another address")
        if at6(1, "recipient") not in (recipient, HANDLER):
            _fail("P6", "the transfer pays another recipient")
    elif selector == DEPOSIT_NATIVE:
        if at6(0, "spoke pool") != chain.spoke_pool:
            _fail("P6", "the deposit goes to another pool")
        if at6(1, "depositor") != wallet:
            _fail("P6", "refunds go to another address")
        if at6(2, "recipient") not in (recipient, HANDLER):
            _fail("P6", "the transfer pays another recipient")
    elif selector in (DEPOSIT, WITHDRAW):
        if recipient != wallet:
            _fail("P6", "wrapping pays the wallet, not the recipient")
    else:
        _fail("P6", "the main call has no recipient rule")


def _encode_order(
    order: dict[str, Any], fields: tuple[tuple[str, FieldType], ...]
) -> str:
    if len(order) != len(fields) or not all(name in order for name, _ in fields):
        _fail("G4", "the order has other fields")
    words = []
    for name, kind in fields:
        v = order[name]
        if kind == "address":
            words.append(_address(v, "G4", f"order.{name}")[2:].rjust(64, "0"))
        elif kind == "bytes32":
            words.append(_bytes32(v, "G4", f"order.{name}")[2:])
        else:
            n = _integer(v, "G4", f"order.{name}")
            if kind == "uint8" and n > 255:
                _fail("G4", f"order.{name} is out of range")
            words.append(format(n, "064x"))
    return "".join(words)


def _gas(
    req: dict[str, Any],
    body: Mapping[str, Any],
    phase1_pay_to: str | None,
    at: _Step,
) -> None:
    if "offer" in req:
        at("G5")
        if (
            phase1_pay_to is not None
            and "payTo" in body
            and _address(body["payTo"], "G5", "payTo") != phase1_pay_to.lower()
        ):
            _fail("G5", "the order pays another address than the offer")
        return

    at("G1")
    pay_chain_id = req.get("payChainId")
    if pay_chain_id is None:
        _fail("G1", "the request has no pay chain")
    chain_id = req.get("chainId")
    if chain_id is None:
        _fail("G1", "the request has no delivery chain")
    pay = _lookup(pay_chain_id)
    if pay is None or pay.gas_pay is None:
        _fail("G1", "the pay chain is not on the allowlist")
    gas_pay = pay.gas_pay
    usdc = pay.usdc
    wallet = _address(req["wallet"], "G1", "wallet") if "wallet" in req else None
    typed = (
        _object(body["typedData"], "G1", "typedData") if "typedData" in body else None
    )
    message = (
        None
        if typed is None
        else _object(typed.get("message"), "G1", "typedData.message")
    )
    if typed is not None and message is not None:
        if typed.get("primaryType") != "ReceiveWithAuthorization":
            _fail("G1", "the authorization is not ReceiveWithAuthorization")
        types = _object(typed.get("types"), "G1", "typedData.types")
        fields = types.get("ReceiveWithAuthorization")
        if not isinstance(fields, list):
            _fail("G1", "the authorization type is missing")
        if len(fields) != len(AUTH_FIELDS) or not all(
            isinstance(f, dict) and f.get("name") == name and f.get("type") == kind
            for f, (name, kind) in zip(fields, AUTH_FIELDS)
        ):
            _fail("G1", "the authorization type has other fields")
        domain = _object(typed.get("domain"), "G1", "typedData.domain")
        if (
            len(domain) != 4
            or domain.get("name") != "USD Coin"
            or domain.get("version") != "2"
        ):
            _fail("G1", "the domain is not USD Coin version 2")
        if not _same_chain(domain.get("chainId"), pay_chain_id):
            _fail("G1", "the domain is on another chain")
        verifying = _address(domain.get("verifyingContract"), "G1", "verifyingContract")
        if verifying != usdc:
            _fail("G1", "the authorization is for another token")
        if wallet is None or _address(message.get("from"), "G1", "message.from") != (
            wallet
        ):
            _fail("G1", "the authorization is from another wallet")
        if _integer(message.get("validAfter"), "G1", "validAfter") != 0:
            _fail("G1", "validAfter is not 0")

    at("G2")
    order = _object(body.get("order"), "G2", "order")
    amount = _units(req.get("amount"), pay.usdc_decimals, "G2", "amount")
    if _integer(order.get("amount"), "G2", "order.amount") != amount:
        _fail("G2", "the order amount is not the requested amount")
    if (
        message is not None
        and _integer(message.get("value"), "G2", "message.value") != amount
    ):
        _fail("G2", "the authorization moves another amount")
    if _integer(order.get("chainId"), "G2", "order.chainId") != int(pay_chain_id):
        _fail("G2", "the order is on another chain")
    if _address(order.get("token"), "G2", "order.token") != usdc:
        _fail("G2", "the order takes another token")
    recipient = _address(req.get("recipient"), "G2", "recipient")
    if _address(order.get("recipient"), "G2", "order.recipient") != recipient:
        _fail("G2", "the order pays another recipient")
    swap = _same(pay_chain_id, chain_id)
    if "refundTo" in req:
        refund_to = _address(req["refundTo"], "G2", "refundTo")
    elif wallet is not None:
        refund_to = wallet
    elif swap:
        refund_to = recipient
    else:
        _fail("G2", "the order has no refund address")
    if _address(order.get("refundTo"), "G2", "order.refundTo") != refund_to:
        _fail("G2", "the order refunds another address")

    at("G3")
    if body.get("kind") != ("swap" if swap else "cross_chain"):
        _fail("G3", "the order is of another kind")
    if not _same(body.get("payChainId"), pay_chain_id) or not _same(
        body.get("chainId"), chain_id
    ):
        _fail("G3", "the offer is for other chains")
    dst = _lookup(chain_id)
    if dst is None:
        _fail("G3", "the delivery chain is not on the allowlist")
    spec = GAS_SWAP if swap else GAS_CROSS_CHAIN
    if _address(body.get("factory"), "G3", "factory") != FACTORY:
        _fail("G3", "the factory is not the pinned one")
    if _address(body.get("template"), "G3", "template") != spec.template:
        _fail("G3", "the template is not the pinned one")

    def pinned(name: str, want: str) -> None:
        if _address(order.get(name), "G3", f"order.{name}") != want:
            _fail("G3", f"order.{name} is not the pinned one")

    if swap:
        pinned("shell", dst.delivery_shell)
    else:
        pinned("router", gas_pay.router_v3)
        pinned("adapter", gas_pay.across_usdc_adapter)
        pinned("handler", HANDLER)
        pinned("dstToken", dst.usdc)
        pinned("dstShell", dst.delivery_shell)
        pinned("dstWrappedNative", dst.wrapped_native)
        if _integer(order.get("dstChainId"), "G3", "order.dstChainId") != int(chain_id):
            _fail("G3", "the order delivers on another chain")
        if _integer(order.get("dstDecimals"), "G3", "order.dstDecimals") != (
            dst.usdc_decimals
        ):
            _fail("G3", "order.dstDecimals is not the pinned one")

    at("G4")
    encoded = _encode_order(order, SWAP_ORDER if swap else ACROSS_ORDER)
    order_hash = _keccak(spec.version_tag + encoded)
    if _bytes32(body.get("orderHash"), "G4", "orderHash") != order_hash:
        _fail("G4", "orderHash does not match the order")
    salted = "0xff" + FACTORY[2:] + order_hash[2:] + spec.init_code_hash[2:]
    pay_to = "0x" + _keccak(salted)[26:]
    if _address(body.get("payTo"), "G4", "payTo") != pay_to:
        _fail("G4", "payTo does not match the order")
    if message is not None:
        if _bytes32(message.get("nonce"), "G4", "message.nonce") != order_hash:
            _fail("G4", "the authorization nonce is not the order hash")
        if _address(message.get("to"), "G4", "message.to") != pay_to:
            _fail("G4", "the authorization pays another address")


def check_plan_swap(
    request: Mapping[str, Any] | BaseModel, body: Mapping[str, Any]
) -> CheckResult:
    return _guard(lambda at: _plan(_request(request), body, at))


def check_get_gas(
    request: Mapping[str, Any] | BaseModel,
    body: Mapping[str, Any],
    phase1_pay_to: str | None = None,
) -> CheckResult:
    return _guard(lambda at: _gas(_request(request), body, phase1_pay_to, at))


def address_refusal(message: str) -> dict[str, Any]:
    return refusal_body("address_not_allowed", message, False)
