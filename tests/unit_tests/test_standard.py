from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_cleave import (
    CleaveFindToken,
    CleaveGetGas,
    CleavePlanSwap,
    CleaveTrackSwap,
)

WALLET = "0x00000000000000000000000000000000000A11cE"


class TestCleavePlanSwapStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return CleavePlanSwap

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "fromChainId": 8453,
            "sellToken": "USDC",
            "buyToken": "WETH",
            "amount": "25",
            "wallet": WALLET,
        }


class TestCleaveTrackSwapStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return CleaveTrackSwap

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"requestId": "g:8453:0x" + "0" * 63 + "1"}


class TestCleaveFindTokenStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return CleaveFindToken

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"chainId": 42161, "query": "USDC"}


class TestCleaveGetGasStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return CleaveGetGas

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "payChainId": 8453,
            "chainId": 8453,
            "recipient": WALLET,
            "amount": "5",
        }
