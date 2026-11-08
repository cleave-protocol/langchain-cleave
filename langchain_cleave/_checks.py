from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from Crypto.Hash import keccak
from pydantic import BaseModel

from langchain_cleave._allowlist import ALLOWLIST
from langchain_cleave._api import refusal_body

CheckRule = Literal["P1", "P2", "P3", "P4", "P5", "P6", "G1", "G2", "G3", "G4"]


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    rule: CheckRule | None = None
    message: str | None = None


@dataclass(frozen=True)
class _GasPay:
    router_v3: str
    across_usdc_adapter: str
    rail_id: str


@dataclass(frozen=True)
class _Target:
    role: str
    selectors: frozenset[str]


@dataclass(frozen=True)
class _Chain:
    usdc: str
    usdc_decimals: int
    wrapped_native: str
    symbols: dict[str, str]
    composer: str
    house_referrer: str
    stablecoins: frozenset[str]
    stargate_rail_id: str
    delivery_shell: str
    gas_pay: _GasPay | None
    targets: dict[str, _Target]
    spoke_pool: str | None
    lanes: dict[int, dict[str, int]]


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


NATIVE = ALLOWLIST["nativeSentinel"].lower()
HANDLER = ALLOWLIST["acrossHandler"].lower()
FACTORY = ALLOWLIST["gas"]["factory"].lower()
FEE_RECIPIENT = ALLOWLIST["gas"]["feeRecipient"].lower()
ZERO_ADDRESS = "0x" + "0" * 40
REFERRAL_BPS = ALLOWLIST["referral"]


def _chain(entry: dict[str, Any]) -> _Chain:
    targets: dict[str, Any] = entry["targets"]
    pay = entry["gasPay"]
    return _Chain(
        usdc=entry["usdc"].lower(),
        usdc_decimals=entry["usdcDecimals"],
        wrapped_native=entry["wrappedNative"].lower(),
        symbols={
            "usdc": entry["usdc"].lower(),
            entry["nativeSymbol"].lower(): NATIVE,
            entry["wrappedSymbol"].lower(): entry["wrappedNative"].lower(),
        },
        composer=entry["composer"].lower(),
        house_referrer=entry["houseReferrer"].lower(),
        stablecoins=frozenset(a.lower() for a in entry["stablecoins"]),
        stargate_rail_id=entry["stargateRailId"].lower(),
        delivery_shell=entry["gasDeliveryShell"].lower(),
        gas_pay=None
        if pay is None
        else _GasPay(
            pay["routerV3"].lower(),
            pay["acrossUsdcAdapter"].lower(),
            pay["railId"].lower(),
        ),
        targets={
            a.lower(): _Target(t["role"], frozenset(t["selectors"]))
            for a, t in targets.items()
        },
        spoke_pool=next(
            (a.lower() for a, t in targets.items() if t["role"] == "across_spoke_pool"),
            None,
        ),
        lanes={
            int(to): {r.lower(): int(d) for r, d in rails.items()}
            for to, rails in entry["lanes"].items()
        },
    )


def _gas_kind(entry: dict[str, Any]) -> _GasKind:
    return _GasKind(
        entry["template"].lower(),
        entry["versionTag"].lower(),
        entry["initCodeHash"].lower(),
    )


CHAINS = {int(k): _chain(v) for k, v in ALLOWLIST["chains"].items()}
GAS_SWAP = _gas_kind(ALLOWLIST["gas"]["swap"])
GAS_CROSS_CHAIN = _gas_kind(ALLOWLIST["gas"]["cross_chain"])
MAX_UINT = (1 << 256) - 1
HALF_UINT = 1 << 255
MAX_SAFE_INTEGER = (1 << 53) - 1
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
HEX_DATA = re.compile(r"0x(?:[0-9a-fA-F]{2})*")
HEX_QUANTITY = re.compile(r"0x[0-9a-fA-F]{1,64}")
BYTES32 = re.compile(r"0x[0-9a-fA-F]{64}")
DECIMAL_INT = re.compile(r"[0-9]{1,78}")
DECIMAL = re.compile(r"([0-9]{1,30})(?:\.([0-9]{1,30}))?")
JS_SPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)

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

SAME_CHAIN = frozenset({SWAP, MEASURED, DEPOSIT, WITHDRAW})
PULLS = {
    SWAP: _Pulls(0, (2,)),
    MEASURED: _Pulls(0, (9,)),
    CROSS: _Pulls(0, (2, 9)),
    BRIDGE: _Pulls(0, (1,)),
    SPOKE_DEPOSIT: _Pulls(2, (4,)),
}
AMOUNT_IN = {
    SWAP: 2,
    MEASURED: 2,
    CROSS: 2,
    CROSS_NATIVE: 2,
    BRIDGE: 1,
    SPOKE_DEPOSIT: 4,
    DEPOSIT_NATIVE: 4,
    WITHDRAW: 0,
}
REFERRAL = {
    SWAP: (6, 7),
    MEASURED: (6, 7),
    CROSS: (6, 7),
    CROSS_NATIVE: (6, 7),
    BRIDGE: (7, 8),
}
ERC20_ONLY = frozenset({CROSS, BRIDGE, SPOKE_DEPOSIT, WITHDRAW})
WETH_AS_ERC20 = {SWAP: 8, MEASURED: 8}
NATIVE_TOKEN = {SWAP: 0, MEASURED: 0, CROSS: 0, CROSS_NATIVE: 0, DEPOSIT_NATIVE: 3}
RAIL_WORD = {CROSS: 12, CROSS_NATIVE: 12, BRIDGE: 2}
GAS_SKEW_SEC = 120
MIN_OUT = {
    SWAP: (3,),
    MEASURED: (3, 10),
    CROSS: (3, 16),
    CROSS_NATIVE: (3, 16),
    BRIDGE: (6,),
    SPOKE_DEPOSIT: (5,),
    DEPOSIT_NATIVE: (6,),
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
    if isinstance(request, BaseModel):
        return request.model_dump(exclude_unset=True)
    return dict(request)


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


def _allowed_chain(v: Any, rule: CheckRule, what: str) -> _Chain:
    chain = CHAINS.get(v) if type(v) is int else None
    if chain is None:
        _fail(rule, f"{what} is not on the allowlist")
    return chain


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


def _normal_decimal(v: Any, rule: CheckRule, what: str) -> str:
    m = DECIMAL.fullmatch(v) if isinstance(v, str) else None
    if m is None:
        _fail(rule, f"{what} is not a decimal amount")
    frac = (m.group(2) or "").rstrip("0")
    return str(int(m.group(1))) + ("." + frac if frac else "")


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


def _lane(chain: _Chain, to_chain: Any) -> dict[str, int] | None:
    return chain.lanes.get(to_chain) if _is_number(to_chain) else None


def _to_dst(amount: int, source: int, target: int) -> int:
    if target >= source:
        return amount * 10 ** (target - source)
    quotient = abs(amount) // 10 ** (source - target)
    return quotient if amount >= 0 else -quotient


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
    from_chain: Any = req.get("fromChainId")
    to_chain: Any = req.get("toChainId", from_chain)
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
    chain = _allowed_chain(from_chain, "P2", "the chain")
    _allowed_chain(to_chain, "P2", "the destination chain")
    target = chain.targets.get(main.to)
    if target is None:
        _fail("P2", "the main call's target is not on the allowlist")
    if selector not in target.selectors:
        _fail("P2", "the main call's function is not allowed on its target")
    if (selector in SAME_CHAIN) != _same(to_chain, from_chain):
        _fail("P2", "the main call does not fit the route")

    at("P3")
    send = _object(body.get("send"), "P3", "send")
    if not _same_chain(send.get("chainId"), from_chain):
        _fail("P3", "send is on another chain")
    token = _address(send.get("token"), "P3", "send.token")
    sell: Any = req.get("sellToken")
    sell = sell.strip(JS_SPACE)
    if ADDRESS.fullmatch(sell):
        pinned: str | None = sell.lower()
    else:
        pinned = chain.symbols.get(sell.lower())
    if pinned is not None and pinned != token:
        _fail("P3", "send.token is not the sell token")
    if _normal_decimal(send.get("amount"), "P3", "send.amount") != _normal_decimal(
        req.get("amount"), "P3", "amount"
    ):
        _fail("P3", "send.amount is not the requested amount")
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
        if amount >= HALF_UINT:
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
        pulled = max(_word_int(main.data, k, "P4", "amount") for k in pulls.pulled)
        if last is not None and pulled != last:
            _fail("P4", "the main call pulls another amount than approved")
        if pulled > 2 * _word_int(main.data, AMOUNT_IN[selector], "P4", "amountIn"):
            _fail("P4", "the main call pulls more than twice its input")
    if native and selector in ERC20_ONLY:
        _fail("P4", "a native sell cannot use this function")
    mode = WETH_AS_ERC20.get(selector)
    if (
        mode is not None
        and native
        and _word_int(main.data, mode, "P4", "native mode") != 0
    ):
        _fail("P4", "the main call spends the wrapped token, not the native coin")
    if (
        selector in (CROSS, CROSS_NATIVE)
        and _word_int(main.data, 8, "P4", "native mode") != 1
    ):
        _fail("P4", "the router swap is not encoded as the backend encodes it")
    wrapped_word = NATIVE_TOKEN.get(selector)
    if (
        native
        and wrapped_word is not None
        and _word_address(main.data, wrapped_word, "P4", "token")
        != chain.wrapped_native
    ):
        _fail("P4", "the main call spends another token")
    if native or token == chain.wrapped_native:
        decimals: int | None = 18
    elif token == chain.usdc:
        decimals = chain.usdc_decimals
    else:
        decimals = None
    in_word = AMOUNT_IN.get(selector)
    if (
        decimals is not None
        and in_word is not None
        and _word_int(main.data, in_word, "P4", "amountIn")
        != _units(req.get("amount"), decimals, "P4", "amount")
    ):
        _fail("P4", "the main call sells another amount than requested")
    for k in MIN_OUT.get(selector, ()):
        if _word_int(main.data, k, "P4", "minimum") == 0:
            _fail("P4", "the main call has no minimum")

    at("P5")
    fees = body.get("fees", [])
    if not isinstance(fees, list):
        _fail("P5", "fees are malformed")
    value = _units(send.get("amount"), 18, "P5", "send.amount") if native else 0
    objects = [_object(f, "P5", "fee") for f in fees]
    messaging = [f for f in objects if f.get("kind") == "native_messaging"]
    if len(messaging) > 1:
        _fail("P5", "more than one native messaging fee")
    for fee in messaging:
        rail_word = RAIL_WORD.get(selector)
        if (
            rail_word is None
            or "0x" + _word(main.data, rail_word, "P5", "railId")
            != chain.stargate_rail_id
        ):
            _fail("P5", "this route takes no native messaging fee")
        value += _units(fee.get("amount"), 18, "P5", "fee amount")
    if main.value != value:
        _fail("P5", "the main call sends another native value")
    referral = REFERRAL.get(selector)
    if referral is not None:
        referrer = _word_address(main.data, referral[0], "P5", "referrer")
        bps = _word_int(main.data, referral[1], "P5", "referral fee")
        same_chain = selector in SAME_CHAIN
        free_bps = REFERRAL_BPS["sameChainBps" if same_chain else "stablecoinBps"]
        free = (ZERO_ADDRESS, int(free_bps))
        house = (chain.house_referrer, int(REFERRAL_BPS["crossChainBps"]))
        dst = _lookup(to_chain)
        receive = body.get("receive")
        shown = receive.get("token") if isinstance(receive, dict) else None
        buy: Any = req.get("buyToken")
        buy = buy.strip(JS_SPACE)
        requested: str | None
        if ADDRESS.fullmatch(buy):
            requested = buy.lower()
        elif dst is None:
            requested = None
        else:
            requested = dst.symbols.get(buy.lower())
        shown_token = (
            shown.lower()
            if isinstance(shown, str) and ADDRESS.fullmatch(shown)
            else None
        )
        if (
            requested is not None
            and shown_token is not None
            and shown_token != requested
        ):
            _fail("P5", "receive.token is not the buy token")
        token_out = requested if requested is not None else shown_token
        if same_chain:
            allowed = [free]
        elif token not in chain.stablecoins:
            allowed = [house]
        elif token_out is None:
            allowed = [free, house]
        else:
            stable_out = dst is not None and token_out in dst.stablecoins
            allowed = [free if stable_out else house]
        if (referrer, bps) not in allowed:
            _fail("P5", "the referral fee goes to another address or has another size")

    at("P6")
    recipient = (
        _address(req["recipient"], "P6", "recipient") if "recipient" in req else wallet
    )
    arrival = None if _same(to_chain, from_chain) else _lookup(to_chain)
    composer = None if arrival is None else arrival.composer

    def arrives(a: str) -> bool:
        return a in (recipient, HANDLER, composer)

    def at6(k: int, what: str) -> str:
        return _word_address(main.data, k, "P6", what)

    def lane(rail_word: int, dst_word: int) -> None:
        rails = _lane(chain, to_chain)
        rail = "0x" + _word(main.data, rail_word, "P6", "railId")
        dst_id = None if rails is None else rails.get(rail)
        if dst_id is None or _word_int(main.data, dst_word, "P6", "destination") != (
            dst_id
        ):
            _fail("P6", "the transfer goes to another chain")

    target_chain = _lookup(to_chain)
    buy_token: Any = req.get("buyToken")
    buy_token = buy_token.strip(JS_SPACE)
    if ADDRESS.fullmatch(buy_token):
        out: str | None = buy_token.lower()
    else:
        out = (
            None
            if target_chain is None
            else target_chain.symbols.get(buy_token.lower())
        )
    if out is None:
        out_word = None
    elif out == NATIVE:
        out_word = None if target_chain is None else target_chain.wrapped_native
    else:
        out_word = out

    def output(k: int) -> None:
        if out_word is not None and at6(k, "output token") != out_word:
            _fail("P6", "the main call delivers another token")

    def destination(k: int) -> None:
        if _word_int(main.data, k, "P6", "destination") != int(to_chain):
            _fail("P6", "the transfer goes to another chain")

    if selector in (SWAP, MEASURED):
        if at6(5, "recipient") != recipient:
            _fail("P6", "the swap pays another recipient")
        output(1)
        if not native:
            flag = _word_int(main.data, 8, "P6", "native output")
            if out is not None:
                wrong = flag != (0 if out == NATIVE else 1)
            else:
                wrong = flag == 0 and at6(1, "output token") != chain.wrapped_native
            if wrong:
                _fail("P6", "the main call delivers another token")
    elif selector in (CROSS, CROSS_NATIVE):
        if at6(5, "source recipient") != main.to:
            _fail("P6", "the source swap pays out of the router")
        if not arrives(at6(14, "recipient")):
            _fail("P6", "the transfer pays another recipient")
        if at6(15, "refund recipient") != wallet:
            _fail("P6", "refunds go to another address")
        lane(12, 13)
    elif selector == BRIDGE:
        if not arrives(at6(4, "recipient")):
            _fail("P6", "the transfer pays another recipient")
        if at6(5, "refund recipient") != wallet:
            _fail("P6", "refunds go to another address")
        lane(2, 3)
    elif selector == SPOKE_DEPOSIT:
        if at6(0, "depositor") != wallet:
            _fail("P6", "refunds go to another address")
        if at6(1, "recipient") not in (recipient, HANDLER):
            _fail("P6", "the transfer pays another recipient")
        destination(6)
        output(3)
    elif selector == DEPOSIT_NATIVE:
        if at6(0, "spoke pool") != chain.spoke_pool:
            _fail("P6", "the deposit goes to another pool")
        if at6(1, "depositor") != wallet:
            _fail("P6", "refunds go to another address")
        if at6(2, "recipient") not in (recipient, HANDLER):
            _fail("P6", "the transfer pays another recipient")
        destination(7)
        output(5)
    elif selector in (DEPOSIT, WITHDRAW):
        if recipient != wallet:
            _fail("P6", "wrapping pays the wallet, not the recipient")
        wrapped = chain.wrapped_native if selector == DEPOSIT else NATIVE
        if out is not None and out != wrapped:
            _fail("P6", "the main call delivers another token")
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


def _gas(req: dict[str, Any], body: Mapping[str, Any], now_sec: int, at: _Step) -> None:
    if req.get("offer") is not None:
        return

    at("G1")
    pay_chain_id = req.get("payChainId")
    if pay_chain_id is None:
        _fail("G1", "the request has no pay chain")
    chain_id = req.get("chainId")
    if chain_id is None:
        _fail("G1", "the request has no delivery chain")
    pay = _allowed_chain(pay_chain_id, "G1", "the pay chain")
    if pay.gas_pay is None:
        _fail("G1", "the pay chain is not on the allowlist")
    _allowed_chain(chain_id, "G1", "the delivery chain")
    gas_pay = pay.gas_pay
    usdc = pay.usdc
    wallet = _address(req["wallet"], "G1", "wallet") if "wallet" in req else None
    if wallet is not None and "typedData" not in body:
        _fail("G1", "the offer has no authorization to sign")
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
    deadline = _integer(order.get("deadline"), "G2", "order.deadline")
    if (
        message is not None
        and _integer(message.get("validBefore"), "G2", "validBefore") > deadline
    ):
        _fail("G2", "the authorization outlives the order")
    window = (
        900 if _same(pay_chain_id, chain_id) and not _same(pay_chain_id, 1) else 1200
    )
    if deadline > int(now_sec + window + GAS_SKEW_SEC):
        _fail("G2", "the order deadline is too far away")
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
    fee = _integer(order.get("fee"), "G2", "order.fee")
    if fee >= amount:
        _fail("G2", "the fee takes the whole amount")
    shown_fee = _object(body.get("fee"), "G2", "fee").get("usdc")
    if fee != _units(shown_fee, pay.usdc_decimals, "G2", "fee.usdc"):
        _fail("G2", "the order fee is not the fee shown")
    min_out = _integer(order.get("minOut"), "G2", "order.minOut")
    if min_out == 0 or min_out != _integer(
        _object(body.get("receive"), "G2", "receive").get("minimumWei"),
        "G2",
        "receive.minimumWei",
    ):
        _fail("G2", "the order minimum is not the minimum shown")
    dst = _lookup(chain_id)
    if dst is None:
        _fail("G2", "the delivery chain is not on the allowlist")
    if not swap:
        relay_fee = _integer(order.get("relayFee"), "G2", "order.relayFee")
        if relay_fee * 2 > amount - fee:
            _fail("G2", "the relay fee is above half the amount")
        output = _integer(order.get("outputAmount"), "G2", "order.outputAmount")
        net = amount - fee - relay_fee
        if output != _to_dst(net, pay.usdc_decimals, dst.usdc_decimals):
            _fail("G2", "the bridged amount is not the amount after fees")

    at("G3")
    if body.get("kind") != ("swap" if swap else "cross_chain"):
        _fail("G3", "the order is of another kind")
    if not _same(body.get("payChainId"), pay_chain_id) or not _same(
        body.get("chainId"), chain_id
    ):
        _fail("G3", "the offer is for other chains")
    spec = GAS_SWAP if swap else GAS_CROSS_CHAIN
    if _address(body.get("factory"), "G3", "factory") != FACTORY:
        _fail("G3", "the factory is not the pinned one")
    if _address(body.get("template"), "G3", "template") != spec.template:
        _fail("G3", "the template is not the pinned one")

    def pinned(name: str, want: str) -> None:
        if _address(order.get(name), "G3", f"order.{name}") != want:
            _fail("G3", f"order.{name} is not the pinned one")

    pinned("feeRecipient", FEE_RECIPIENT)
    if _integer(order.get("maxProtocolFeeBps"), "G3", "order.maxProtocolFeeBps") != 0:
        _fail("G3", "the order allows a protocol fee")
    if swap:
        pinned("shell", dst.delivery_shell)
    else:
        pinned("router", gas_pay.router_v3)
        pinned("adapter", gas_pay.across_usdc_adapter)
        pinned("handler", HANDLER)
        pinned("dstToken", dst.usdc)
        pinned("dstShell", dst.delivery_shell)
        pinned("dstWrappedNative", dst.wrapped_native)
        rail_id = _bytes32(order.get("railId"), "G3", "order.railId")
        rails = _lane(pay, chain_id)
        if rail_id != gas_pay.rail_id or rails is None or rail_id not in rails:
            _fail("G3", "order.railId is not the pinned one")
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
    now_sec: int | None = None,
) -> CheckResult:
    now = int(time.time()) if now_sec is None else now_sec
    return _guard(lambda at: _gas(_request(request), body, now, at))


PLAN_KEYS = (
    "requestId",
    "blocked",
    "requiresConfirmation",
    "route",
    "send",
    "pullsAtMost",
    "receive",
    "fees",
    "notices",
    "quotedAt",
    "validUntil",
    "calls",
    "next",
)
CALL_KEYS = ("chainId", "from", "to", "data", "value")
OFFER_KEYS = (
    "offer",
    "kind",
    "payChainId",
    "chainId",
    "payTo",
    "factory",
    "template",
    "order",
    "blob",
    "hubs",
    "message",
    "orderHash",
    "outcome",
    "pay",
    "fee",
    "receive",
    "deadline",
    "refundTo",
    "typedData",
    "etaSec",
    "verify",
)
SUBMIT_KEYS = ("requestId", "state", "payTo", "deadline")
ERROR_KEYS = (
    "code",
    "reason",
    "message",
    "detail",
    "retryable",
    "retryAfterSec",
    "suggestion",
    "notices",
)
DOMAIN_KEYS = ("name", "version", "chainId", "verifyingContract")
MESSAGE_KEYS = tuple(name for name, _ in AUTH_FIELDS)


def _pick(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: source[k] for k in keys if k in source}


def api_refusal(body: Mapping[str, Any]) -> dict[str, Any] | None:
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    return {
        **_pick(body, ("apiVersion", "requestId")),
        "error": _pick(error, ERROR_KEYS),
    }


def address_refusal(message: str) -> dict[str, Any]:
    return refusal_body("address_not_allowed", message, False)


GAS_MIN = 21_000
GAS_MAX = 33_554_432


def _is_gas_limit(v: Any) -> bool:
    if not (isinstance(v, str) and HEX_QUANTITY.fullmatch(v) and v[2] != "0"):
        return False
    return GAS_MIN <= int(v, 16) <= GAS_MAX


def keep_plan(body: Mapping[str, Any]) -> dict[str, Any]:
    out = _pick(body, PLAN_KEYS)
    calls = body["calls"]
    kept = [_pick(c, CALL_KEYS) for c in calls]
    if kept and _is_gas_limit(calls[-1].get("gas")):
        kept[-1]["gas"] = calls[-1]["gas"]
    out["calls"] = kept
    return out


def keep_gas(
    request: Mapping[str, Any] | BaseModel, body: Mapping[str, Any]
) -> dict[str, Any]:
    if _request(request).get("offer") is not None:
        return _pick(body, SUBMIT_KEYS)
    out = _pick(body, OFFER_KEYS)
    typed = body.get("typedData")
    if isinstance(typed, dict):
        out["typedData"] = {
            "domain": _pick(typed["domain"], DOMAIN_KEYS),
            "types": {
                "ReceiveWithAuthorization": [
                    _pick(f, ("name", "type"))
                    for f in typed["types"]["ReceiveWithAuthorization"]
                ]
            },
            "primaryType": typed["primaryType"],
            "message": _pick(typed["message"], MESSAGE_KEYS),
        }
    return out


def screen_plan_swap(
    request: Mapping[str, Any] | BaseModel, body: Mapping[str, Any]
) -> dict[str, Any]:
    refusal = api_refusal(body)
    if refusal is not None:
        return refusal
    result = check_plan_swap(request, body)
    if not result.ok:
        return address_refusal(str(result.message))
    return keep_plan(body)


def screen_get_gas(
    request: Mapping[str, Any] | BaseModel,
    body: Mapping[str, Any],
    now_sec: int | None = None,
) -> dict[str, Any]:
    refusal = api_refusal(body)
    if refusal is not None:
        return refusal
    result = check_get_gas(request, body, now_sec)
    if not result.ok:
        return address_refusal(str(result.message))
    return keep_gas(request, body)
