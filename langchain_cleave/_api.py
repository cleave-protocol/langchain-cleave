from __future__ import annotations

import json
from functools import cache
from importlib import metadata
from typing import Any

import httpx
from langchain_core.tools import ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_BASE_URL = "https://api.cleave.fi"
DEFAULT_TIMEOUT = 60.0
TOOL_PATHS = {
    "plan_swap": "/agent/v1/plan_swap",
    "track_swap": "/agent/v1/track_swap",
    "find_token": "/agent/v1/find_token",
    "get_gas": "/agent/v1/get_gas",
}


@cache
def _version() -> str:
    try:
        return metadata.version("langchain-cleave")
    except metadata.PackageNotFoundError:
        return "0.0.0"


class CleaveAPIError(ToolException):
    def __init__(self, status: int | None, body: dict[str, Any]) -> None:
        self.status = status
        self.body = body
        found = body.get("error")
        error: dict[str, Any] = found if isinstance(found, dict) else {}
        self.code: str | None = error.get("code")
        self.reason: str | None = error.get("reason")
        self.retryable: bool = bool(error.get("retryable", False))
        super().__init__(json.dumps(body, ensure_ascii=False))


def refusal_body(
    code: str, message: str, retryable: bool, detail: str | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        error["detail"] = detail
    error["retryable"] = retryable
    return {"error": error}


def _refusal(
    status: int | None, code: str, message: str, retryable: bool
) -> CleaveAPIError:
    return CleaveAPIError(status, refusal_body(code, message, retryable))


def _network(err: httpx.HTTPError) -> CleaveAPIError:
    return _refusal(None, "network_error", str(err) or type(err).__name__, True)


class CleaveAPIWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("CLEAVE_API_KEY", default=None)
    )
    client: httpx.Client | None = Field(default=None, exclude=True)
    async_client: httpx.AsyncClient | None = Field(default=None, exclude=True)

    def _url(self, tool: str) -> str:
        return self.base_url.rstrip("/") + TOOL_PATHS[tool]

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"langchain-cleave/{_version()}",
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    @staticmethod
    def _answer(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.is_success:
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                raise CleaveAPIError(response.status_code, body)
            if isinstance(body, dict):
                return body
            raise _refusal(
                response.status_code,
                "bad_response",
                "The answer is not a JSON object",
                True,
            )
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            raise CleaveAPIError(response.status_code, body)
        raise _refusal(
            response.status_code,
            "http_error",
            f"HTTP {response.status_code}",
            response.status_code >= 500 or response.status_code == 429,
        )

    def _request(
        self,
        client: httpx.Client | httpx.AsyncClient,
        tool: str,
        payload: dict[str, Any],
    ) -> httpx.Request:
        request = client.build_request(
            "POST",
            self._url(tool),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if self.api_key is None:
            request.headers.pop("Authorization", None)
        return request

    def call(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if self.client is not None:
                request = self._request(self.client, tool, payload)
                response = self.client.send(request, auth=None)
            else:
                with httpx.Client() as client:
                    request = self._request(client, tool, payload)
                    response = client.send(request, auth=None)
        except httpx.HTTPError as err:
            raise _network(err) from err
        return self._answer(response)

    async def acall(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if self.async_client is not None:
                request = self._request(self.async_client, tool, payload)
                response = await self.async_client.send(request, auth=None)
            else:
                async with httpx.AsyncClient() as client:
                    request = self._request(client, tool, payload)
                    response = await client.send(request, auth=None)
        except httpx.HTTPError as err:
            raise _network(err) from err
        return self._answer(response)
