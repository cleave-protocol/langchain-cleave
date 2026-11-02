# langchain-cleave

LangChain tools for [Cleave](https://cleave.fi), a DEX aggregator on Ethereum,
Arbitrum, Base, BNB Chain, Polygon and Optimism, with split routing and
one-step cross-chain swaps.

The package gives a LangChain agent the four tools of Cleave's MCP server,
with the same names, inputs and answers. They call the keyless
[Agent REST API](https://cleave.fi/docs/agents/rest). No API key is needed.

The tools return unsigned calls for the user's wallet to sign and send. They
never take a private key. They sign nothing; the only thing they send is a gas
order the wallet has signed, and only when the gas relay is turned on (it is
off by default; see [Gas top-ups](#gas-top-ups)).

## Installation

```bash
pip install -U langchain-cleave
```

```bash
uv add langchain-cleave
```

Python 3.10 or later.

## Tools

| Tool | Class | What it does |
| --- | --- | --- |
| `cleave_plan_swap` | `CleavePlanSwap` | Plans a swap on one chain, a cross-chain swap or a bridge, and returns unsigned calls in signing order |
| `cleave_track_swap` | `CleaveTrackSwap` | Reports the state of a sent plan or a gas order, and what arrived |
| `cleave_find_token` | `CleaveFindToken` | Finds a token by symbol or address and flags look-alikes |
| `cleave_get_gas` | `CleaveGetGas` | Offers native gas for USDC; with the relay turned on, passes the order the wallet signed to Cleave's relayer |

`CleaveToolkit().get_tools()` returns all four.

Every input field is the same as the MCP tool's: chain ids are integers
(1, 10, 56, 137, 8453, 42161), amounts are decimal strings in token units,
and tokens are listed symbols or contract addresses.

## Use with an agent

This example also needs `langchain` and a chat model package:

```bash
pip install -U langchain-cleave langchain "langchain[anthropic]"
```

```python
from langchain.agents import create_agent
from langchain_cleave import CleaveToolkit

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=CleaveToolkit().get_tools(),
)

agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Plan a swap of 25 USDC to WETH on Base for wallet "
                    "0x00000000000000000000000000000000000A11cE"
                ),
            }
        ]
    }
)
```

## Call a tool directly

```python
from langchain_cleave import CleaveFindToken, CleavePlanSwap

CleaveFindToken().invoke({"chainId": 42161, "query": "USDC"})

plan = CleavePlanSwap().invoke(
    {
        "fromChainId": 8453,
        "sellToken": "USDC",
        "buyToken": "WETH",
        "amount": "25",
        "wallet": "0x00000000000000000000000000000000000A11cE",
    }
)
```

A plan is a dict. Read it in this order:

1. `blocked` and `notices`: a `block` notice means `calls` is empty.
2. `requiresConfirmation`: set by a `warn` notice or a recipient other than
   the wallet. Show the plan to the user before signing.
3. `route.outcomeKind` and `receive`: what arrives, the minimum, and the
   fallback or refund.
4. `fees`: costs charged by others, such as a bridge fee. Cleave's fees are
   included in the price. Network gas is not listed.
5. `validUntil`: sign and send `calls` in order before this unix time.

### Before anything is signed

- Do not act on a blocked plan.
- Show the user the warnings, the recipient, the amounts and the slippage,
  and get their confirmation before any wallet signs. The `cleave_plan_swap`
  description tells the model the same, and not to raise slippage above the
  default 50 bps without the user's consent.
- Never change the recipient or raise slippage to get past a refusal or a
  warning.
- `blocked`, `requiresConfirmation` and the notices are advisory. They bind
  only if your application, or the signer it uses, enforces them.

### Signing and tracking

- `calls` are in signing order. Any approval calls come first; the last call
  is the swap or transfer itself.
- Each approval must be mined and succeed before the next call is sent.
- If `validUntil` passes before the last call is sent, plan again and sign
  the new calls.
- Track the mined hash of the last call, the execution transaction, with
  `cleave_track_swap`, passing `chainId`, `txHash` and the plan's `requestId`.
  Do not track an approval's hash: a refusal on an approval hash, such as
  `not_a_cleave_tx`, says nothing about the swap.
- Track before sending a plan again: a plan still in flight that is sent
  twice can pay twice.

### Finding tokens

`cleave_find_token` returns up to ten candidates. When a symbol is shared by
many tokens, the one the user means may not be among them: prefer a verified
contract address. A search by symbol or name is limited to 48 characters
after normalization, although the input schema accepts 64.

## Address checks

Before a `cleave_plan_swap` plan or a `cleave_get_gas` offer reaches the
model, the package checks it against per-chain contract addresses, rails
and tokens pinned in this package:

- `cleave_plan_swap`: every call is on the planned chain and from the
  wallet. The last call goes to a listed contract and function that fits
  the route, sends to the planned chain and pays the requested recipient
  where its calldata names one; refunds go to the wallet. Any call before
  it is an approve of the sell token to that contract for exactly what the
  last call pulls. The amount sold is the requested amount, and the native
  value is what that amount and the listed native messaging fee need.
- `cleave_get_gas`: the authorization is a USDC `ReceiveWithAuthorization`
  from the wallet on the paying chain. The order carries the requested
  amount, recipient and refund address, the pinned contracts, rail and fee
  recipient, and the fee and amounts the offer shows. The order hash and
  the payment address are recomputed and must match what is signed.

A plan or offer that fails a check is not passed on; the tool returns only
this refusal (see [Errors](#errors)):

```json
{
  "error": {
    "code": "address_not_allowed",
    "message": "P2: the main call's target is not on the allowlist",
    "retryable": false
  }
}
```

A plan or offer that passes is returned with only its documented fields.

What these checks do not cover:

- The price, the route and the amount received.
- On routes that swap on arrival or pay a fee through the bridge message,
  the funds go to the Across handler or Cleave's composer, and that
  message decides who receives them. These checks do not read it, so a
  compromised API could redirect those funds. Have a person check such a
  plan before signing.
- When `sellToken` is a symbol other than USDC or the chain's native or
  wrapped native coin, the API resolves it to an address. Check
  `send.token` before signing.
- `next` and the notices are text from the API.

After Cleave redeploys a contract, plans that use it fail these checks
until a package update pins the new address.

`check_plan_swap` and `check_get_gas` are exported for agents that call the
API directly. Each returns a `CheckResult` with `ok`, `rule` and `message`.

Do not pair these tools with a wallet that signs without a human looking.

## Gas top-ups

`cleave_get_gas` takes two calls with separate fields; fields from one call
are refused in the other.

1. Offer: `payChainId`, `chainId`, `recipient`, `amount` (USDC), `wallet`
   and optionally `refundTo`. It returns the offer, the fee, the minimum
   native amount, the deadline, the order and, only when `wallet` is given,
   `typedData`: the USDC authorization to sign.
2. Submit: `offer`, `signature` and `wallet`. It passes the order the wallet
   signed to Cleave's relayer, which executes it, and returns a `requestId`
   for `cleave_track_swap`.

```python
from langchain_cleave import CleaveGetGas

offer = CleaveGetGas().invoke(
    {
        "payChainId": 8453,
        "chainId": 8453,
        "recipient": "0x00000000000000000000000000000000000A11cE",
        "amount": "5",
        "wallet": "0x00000000000000000000000000000000000A11cE",
    }
)
```

The refund address in the order is `refundTo` if given, else `wallet`, else
the `recipient` for an order on one chain. An order paid on another chain
needs `wallet` or `refundTo`. Adding `wallet` only at submission does not
change the refund address of the order already offered. Review both the
recipient and the refund address in `order` before signing.

| Check | Value |
| --- | --- |
| Amount | USDC, at most 6 decimals |
| Smallest order | 2 USDC, or 5 USDC when the gas is delivered on Ethereum |
| Largest order | 50 USDC |
| USDC paid on | Ethereum, Arbitrum, Base, Optimism, Polygon |

- An authorization that is never submitted never debits the wallet.
- Once an order is funded, its outcome is whatever `cleave_track_swap`
  reports for `g:<payChainId>:<orderHash>`. Do not assume a refund because a
  deadline passed; track until the status shows it.
- See [x402 gas top-up](https://cleave.fi/docs/agents/gas).

### The relay is off by default

The submit call is off by default: the tool returns offers, and a call with
`offer` or `signature` comes back as a refusal with code `relay_disabled`,
without a request. Turn it on in code, not from the agent:

```python
from langchain_cleave import CleaveGetGas, CleaveToolkit

tools = CleaveToolkit(include_gas_relay=True).get_tools()
gas = CleaveGetGas(allow_relay=True)
```

With the relay on, an agent can keep a person in the loop before each signed
order is sent. `HumanInTheLoopMiddleware` pauses on that tool; the agent needs
a checkpointer:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=CleaveToolkit(include_gas_relay=True).get_tools(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"cleave_get_gas": True})],
    checkpointer=InMemorySaver(),
)
```

### When a submission fails

A submission that times out, loses its connection, or gets a server error or
an unreadable answer may still have been accepted. The tool then returns
`outcome_unknown`, which is not retryable. When the tool returned the offer
itself, the body carries `payChainId`, `orderHash` and `trackRequestId`.

- Track `g:<payChainId>:<orderHash>` with `cleave_track_swap` before signing
  a new offer. Do not sign a replacement until the original shows its
  outcome: two signed orders can both be paid.
- Cancelling the call, or the agent run, does not cancel an order the relayer
  already accepted.
- A later `authorization_used` or `offer_expired` refusal does not prove that
  an earlier submission failed. Track the order.

## Errors

What a tool returns depends on how it is called:

| Call | Success | Refusal |
| --- | --- | --- |
| `tool.invoke({...})` | a `dict` | a JSON string |
| `tool.invoke(tool_call)` | a `ToolMessage` | a `ToolMessage` with `status="error"` |

A refusal from the API carries the API's JSON body, as over MCP:
`{"apiVersion", "requestId", "error": {"code", "reason", "message", "retryable", ...}}`.
From `cleave_plan_swap` and `cleave_get_gas`, a refusal from the API comes
back with only its `apiVersion`, `requestId` and `error` fields.
Refusals made by the package itself have only `error`:

- `network_error`: the request did not complete (retryable).
- `invalid_param`: the input breaks the schema, such as an unsupported or
  boolean chain id, a malformed address or an explicit `null`. No request is
  sent; `detail` names each field. Omit an optional field instead of sending
  `null`.
- `address_not_allowed`: the answer failed a check, and nothing else from it
  is returned; see [Address checks](#address-checks).
- `relay_disabled` and `outcome_unknown`: see [Gas top-ups](#gas-top-ups).

To raise instead, pass `handle_tool_error=False`. The tool then raises
`CleaveAPIError`, a `ToolException` with `status`, `code`, `reason`,
`retryable` and the full `body`. Pass `handle_validation_error=False` to raise
a pydantic `ValidationError` for schema violations.
## Configuration

```python
import httpx
from langchain_cleave import CleaveAPIWrapper, CleaveToolkit

api = CleaveAPIWrapper(
    base_url="https://api.cleave.fi",
    timeout=60.0,
    client=httpx.Client(timeout=60.0),
)
tools = CleaveToolkit(api_wrapper=api).get_tools()
```

- `base_url`: defaults to `https://api.cleave.fi`.
- `timeout`: the httpx timeout in seconds, applied to each phase of a
  request (connect, read, write and pool), 60 by default. It is passed on
  every request, also with your own client.
- `client`, `async_client`: optional `httpx` clients, for proxies or reuse.
  The package sets its own authentication on each request: an
  `Authorization` header or `auth` set on your client is never sent.
- `api_key`: optional partner key, of the form `cq_live_<key>`. It is sent as
  `Authorization: Bearer cq_live_<key>`. Keyless use needs none.
  - When the argument is omitted, the key is read from `CLEAVE_API_KEY`.
  - `api_key=None` means keyless, even when `CLEAVE_API_KEY` is set.
  - A set key replaces keyless use: every call is checked against that key,
    and a bad or revoked key fails every call. There is no fallback to
    keyless.

Call the API from a server, bot or agent process. Requests that carry an
`Origin` header, as browsers send, are refused.

## Limits

- Without a key, each caller's address can make 40 requests per 10 seconds
  and 120 per minute, and all keyless agents share one planning capacity.
  When it is busy, the refusal is `rate_limited` with a short
  `retryAfterSec`.
- The token to receive must be on Cleave's token list.
- A cross-chain plan from one token to another refuses `maxSlippageBps: 0`.
- The schema lists six chains, but a deployment can turn a chain off; a plan
  on it is then refused with `unsupported_chain`.

## Links

- [Cleave for agents](https://cleave.fi/docs/agents)
- [Agent REST API](https://cleave.fi/docs/agents/rest)
- [OpenAPI](https://api.cleave.fi/agent/v1/openapi.json)
- [MCP server](https://cleave.fi/docs/agents/mcp): `https://api.cleave.fi/mcp`

## Terms

Use of the API is subject to Cleave's [Terms of Use](https://cleave.fi/terms)
and [Privacy Policy](https://cleave.fi/privacy).

## License

MIT
