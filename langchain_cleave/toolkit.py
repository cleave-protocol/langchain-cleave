from __future__ import annotations

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field

from langchain_cleave._api import CleaveAPIWrapper
from langchain_cleave.tools import (
    CleaveFindToken,
    CleaveGetGas,
    CleavePlanSwap,
    CleaveTrackSwap,
)


class CleaveToolkit(BaseToolkit):
    api_wrapper: CleaveAPIWrapper = Field(default_factory=CleaveAPIWrapper)
    include_gas_relay: bool = False

    def get_tools(self) -> list[BaseTool]:
        return [
            CleavePlanSwap(api_wrapper=self.api_wrapper),
            CleaveTrackSwap(api_wrapper=self.api_wrapper),
            CleaveFindToken(api_wrapper=self.api_wrapper),
            CleaveGetGas(
                api_wrapper=self.api_wrapper, allow_relay=self.include_gas_relay
            ),
        ]
