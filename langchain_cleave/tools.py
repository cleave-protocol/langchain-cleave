from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    ValidationError,
    model_validator,
)

from langchain_cleave import _descriptions
from langchain_cleave._api import CleaveAPIError, CleaveAPIWrapper, refusal_body
from langchain_cleave._checks import (
    CheckResult,
    address_refusal,
    api_refusal,
    check_get_gas,
    check_plan_swap,
    keep_gas,
    keep_plan,
)


def _not_bool(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("a chain id is an integer, not a boolean")
    return value


ChainId = Annotated[Literal[1, 10, 56, 137, 8453, 42161], BeforeValidator(_not_bool)]
ADDRESS = r"^0x[0-9a-fA-F]{40}$"
HASH = r"^0x[0-9a-fA-F]{64}$"
AMOUNT = r"^[0-9]+(\.[0-9]+)?$"
REQUEST_ID = r"^([A-Za-z0-9-]{1,64}|g:[0-9]{1,10}:0x[0-9a-fA-F]{64})$"
SIGNATURE = r"^0x[0-9a-fA-F]+$"


def _no_default(schema: dict[str, Any]) -> None:
    schema.pop("default", None)


def _sent(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def omittable(**kwargs: Any) -> Any:
    return Field(default=None, json_schema_extra=_no_default, **kwargs)


class PlanSwapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fromChainId: ChainId = Field(description="Chain id of the token to sell")
    toChainId: ChainId = omittable(
        description="Chain id of the token to receive; defaults to fromChainId",
    )
    sellToken: str = Field(
        min_length=1,
        max_length=64,
        description="A listed symbol or a contract address",
    )
    buyToken: str = Field(
        min_length=1,
        max_length=64,
        description="A listed symbol or a contract address",
    )
    amount: str = Field(
        pattern=AMOUNT,
        description='Decimal amount of sellToken, in token units, such as "25.5"',
    )
    wallet: str = Field(pattern=ADDRESS, description="The wallet that signs and pays")
    recipient: str = omittable(
        pattern=ADDRESS,
        description="Who receives; defaults to wallet",
    )
    maxSlippageBps: StrictInt = omittable(
        ge=0,
        le=5000,
        description="Slippage limit in basis points, 0 to 5000",
    )


class TrackSwapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chainId: ChainId = omittable(description="Chain id of the sent transaction")
    txHash: str = omittable(pattern=HASH, description="Hash of the sent transaction")
    requestId: str = omittable(
        min_length=1,
        max_length=79,
        pattern=REQUEST_ID,
        description=(
            "The requestId of the plan, when known; for a gas order, the "
            "requestId alone"
        ),
    )


class FindTokenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chainId: ChainId = Field(description="Chain id to search on")
    query: str = Field(
        min_length=1,
        max_length=64,
        description="A symbol or a contract address",
    )


class GetGasInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payChainId: ChainId = omittable(description="Chain id the USDC is paid on")
    chainId: ChainId = omittable(description="Chain id the native coin is for")
    recipient: str = omittable(
        pattern=ADDRESS, description="Who receives the native coin"
    )
    amount: str = omittable(
        pattern=AMOUNT,
        description='USDC to pay, as a decimal string such as "5"',
    )
    wallet: str = omittable(
        pattern=ADDRESS,
        description=(
            "The paying wallet; with it, the offer carries the authorization "
            "to sign. Required with offer and signature"
        ),
    )
    refundTo: str = omittable(
        pattern=ADDRESS,
        description=(
            "The refund address on the pay chain; an order from another chain "
            "needs wallet or refundTo"
        ),
    )
    offer: str = omittable(
        min_length=1,
        max_length=4096,
        description="The offer string from the first call, sent back with signature",
    )
    signature: str = omittable(
        pattern=SIGNATURE,
        description="The wallet's signature of the offer's USDC authorization",
    )


INVALID_MESSAGE = "Some details of this request are not valid."
PLAN_CONFIRM = (
    " Before any wallet signs, show the user the warnings, recipient, amounts "
    "and slippage, and get their confirmation. Do not raise slippage above the "
    "default 50 bps without the user's consent."
)
GAS_SUBMIT = (
    " If a submission fails or times out, the order may still have been "
    "accepted: track g:<payChainId>:<orderHash> before signing a new offer."
)
OFFERS_KEPT = 64
RELAY_OFF = (
    " In this setup the second call is turned off, so the tool only returns "
    "offers; a person must turn it on before a signed order can be passed to "
    "Cleave's relayer."
)


def refusal_content(error: ToolException) -> str:
    return str(error)


def screened(
    answer: dict[str, Any],
    check: Callable[[], CheckResult],
    keep: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    refusal = api_refusal(answer)
    if refusal is not None:
        return refusal
    result = check()
    if not result.ok:
        raise CleaveAPIError(None, address_refusal(str(result.message)))
    return keep()


def api_error(err: CleaveAPIError) -> CleaveAPIError:
    return CleaveAPIError(err.status, api_refusal(err.body) or err.body)


def invalid_param_content(error: ValidationError) -> str:
    problems = []
    for e in error.errors():
        where = ".".join(str(part) for part in e.get("loc", ())) or "input"
        problems.append(f"{where}: {e.get('msg', 'not valid')}")
    body = refusal_body("invalid_param", INVALID_MESSAGE, False, "; ".join(problems))
    return json.dumps(body, ensure_ascii=False)


class _CleaveTool(BaseTool):
    endpoint: ClassVar[str]

    api_wrapper: CleaveAPIWrapper = Field(default_factory=CleaveAPIWrapper)
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = (
        refusal_content
    )
    handle_validation_error: bool | str | Callable[..., str] | None = (
        invalid_param_content
    )

    @property
    def tool_call_schema(self) -> type[BaseModel]:
        schema = self.args_schema
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("args_schema must be a pydantic model")
        return schema

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _sent(kwargs)
        return self.api_wrapper.call(self.endpoint, payload)

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _sent(kwargs)
        return await self.api_wrapper.acall(self.endpoint, payload)


class CleavePlanSwap(_CleaveTool):
    endpoint: ClassVar[str] = "plan_swap"
    name: str = "cleave_plan_swap"
    description: str = _descriptions.PLAN_SWAP + PLAN_CONFIRM
    args_schema: type[BaseModel] = PlanSwapInput

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            answer = super()._run(run_manager=run_manager, **kwargs)
        except CleaveAPIError as err:
            raise api_error(err) from err
        return screened(
            answer,
            lambda: check_plan_swap(_sent(kwargs), answer),
            lambda: keep_plan(answer),
        )

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            answer = await super()._arun(run_manager=run_manager, **kwargs)
        except CleaveAPIError as err:
            raise api_error(err) from err
        return screened(
            answer,
            lambda: check_plan_swap(_sent(kwargs), answer),
            lambda: keep_plan(answer),
        )


class CleaveTrackSwap(_CleaveTool):
    endpoint: ClassVar[str] = "track_swap"
    name: str = "cleave_track_swap"
    description: str = _descriptions.TRACK_SWAP
    args_schema: type[BaseModel] = TrackSwapInput


class CleaveFindToken(_CleaveTool):
    endpoint: ClassVar[str] = "find_token"
    name: str = "cleave_find_token"
    description: str = _descriptions.FIND_TOKEN
    args_schema: type[BaseModel] = FindTokenInput


class CleaveGetGas(_CleaveTool):
    endpoint: ClassVar[str] = "get_gas"
    name: str = "cleave_get_gas"
    description: str = _descriptions.GET_GAS + GAS_SUBMIT
    args_schema: type[BaseModel] = GetGasInput
    allow_relay: bool = False
    _offers: OrderedDict[str, tuple[int, str]] = PrivateAttr(
        default_factory=OrderedDict
    )

    @model_validator(mode="before")
    @classmethod
    def _relay_note(cls, values: Any) -> Any:
        if (
            isinstance(values, dict)
            and "description" not in values
            and not values.get("allow_relay", False)
        ):
            text = _descriptions.GET_GAS + GAS_SUBMIT + RELAY_OFF
            return {**values, "description": text}
        return values

    def _remember(self, answer: dict[str, Any]) -> None:
        offer = answer.get("offer")
        pay = answer.get("payChainId")
        order_hash = answer.get("orderHash")
        if (
            isinstance(offer, str)
            and isinstance(pay, int)
            and not isinstance(pay, bool)
            and isinstance(order_hash, str)
        ):
            self._offers[offer] = (pay, order_hash)
            self._offers.move_to_end(offer)
            while len(self._offers) > OFFERS_KEPT:
                self._offers.popitem(last=False)

    @staticmethod
    def _ambiguous(err: CleaveAPIError) -> bool:
        if err.code in {"network_error", "bad_response"}:
            return True
        return err.code == "http_error" and (err.status or 0) >= 500

    def _unknown(self, offer: str, err: CleaveAPIError) -> CleaveAPIError:
        known = self._offers.get(offer)
        track = (
            "g:<payChainId>:<orderHash>"
            if known is None
            else f"g:{known[0]}:{known[1]}"
        )
        cause = err.body.get("error")
        error: dict[str, Any] = {
            "code": "outcome_unknown",
            "message": (
                "The signed gas order may have been accepted. Track "
                f"{track} with cleave_track_swap before signing a new offer, and "
                "do not sign a replacement until it shows the outcome."
            ),
        }
        if isinstance(cause, dict) and isinstance(cause.get("message"), str):
            error["detail"] = cause["message"]
        error["retryable"] = False
        if known is not None:
            error["payChainId"] = known[0]
            error["orderHash"] = known[1]
            error["trackRequestId"] = track
        return CleaveAPIError(err.status, {"error": error})

    def _relay_check(self, kwargs: dict[str, Any]) -> None:
        if self.allow_relay:
            return
        if kwargs.get("offer") is not None or kwargs.get("signature") is not None:
            raise CleaveAPIError(
                None,
                refusal_body(
                    "relay_disabled",
                    "Passing a signed gas order to Cleave's relayer is turned off "
                    "for this tool. A person must turn it on with "
                    "CleaveGetGas(allow_relay=True) or "
                    "CleaveToolkit(include_gas_relay=True).",
                    False,
                ),
            )

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._relay_check(kwargs)
        offer = kwargs.get("offer")
        try:
            answer = super()._run(run_manager=run_manager, **kwargs)
        except CleaveAPIError as err:
            if offer is not None and self._ambiguous(err):
                raise self._unknown(offer, err) from err
            raise api_error(err) from err
        answer = screened(
            answer,
            lambda: check_get_gas(_sent(kwargs), answer),
            lambda: keep_gas(_sent(kwargs), answer),
        )
        if offer is None:
            self._remember(answer)
        return answer

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._relay_check(kwargs)
        offer = kwargs.get("offer")
        try:
            answer = await super()._arun(run_manager=run_manager, **kwargs)
        except CleaveAPIError as err:
            if offer is not None and self._ambiguous(err):
                raise self._unknown(offer, err) from err
            raise api_error(err) from err
        answer = screened(
            answer,
            lambda: check_get_gas(_sent(kwargs), answer),
            lambda: keep_gas(_sent(kwargs), answer),
        )
        if offer is None:
            self._remember(answer)
        return answer
