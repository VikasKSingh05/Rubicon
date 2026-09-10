# Phase 0 — Teammate Account & Key Setup Checklist

> Each teammate provisions their **own** accounts/keys. Do **not** share credentials across
> teammates. Never commit secrets — `.env` is gitignored from day one.

## 1. MongoDB Atlas (free M0 cluster)
- [ ] Create account at https://www.mongodb.com/cloud/atlas
- [ ] Create a free **M0** cluster
- [ ] Create a database user + allow your IP in **Network Access**
- [ ] Copy the connection string → `MONGO_URI` in `.env`

## 2. Pinata (IPFS) — free tier
- [ ] Create account at https://app.pinata.cloud
- [ ] Generate an API **JWT** (`app.pinata.cloud` → API Keys)
- [ ] Copy JWT → `PINATA_JWT` in `.env` (CIDs are computed locally regardless; pinning needs the JWT)

## 3. Fresh testnet-only MetaMask wallet + faucet POL
- [ ] **Create a brand-new MetaMask wallet dedicated to this project only.**
      🔴 Never reuse a personal/mainnet wallet or key.
- [ ] Add the **Polygon Amoy** testnet to MetaMask:
      - RPC: `https://rpc-amoy.polygon.technology` (chain ID **80002**, symbol **POL**)
      - 🟡 For reliability under load, get a free dedicated endpoint from
        **Alchemy** (https://alchemy.com) or **Infura** → `AMOY_RPC_URL`
- [ ] Fund with faucet POL (~0.1 POL/day). Set this up **now**, not the week you deploy:
      - Alchemy Amoy faucet: https://faucet.alchemy.com (some faucets check mainnet activity)
- [ ] Copy the wallet private key → `DEPLOYER_PRIVATE_KEY` in `.env` (**never commit**)
- [ ] After Phase 5 deployment, set `CONTRACT_ADDRESS` in `.env`

## 4. LLM API key (configurable — see Phase 0 decision)
We support **three** providers behind `LLM_PROVIDER` (`anthropic` | `openai` | `openrouter`), or empty for the deterministic tool-grounded fallback agent.

- **[ ] Anthropic** → https://console.anthropic.com → `ANTHROPIC_API_KEY`
  - model: `AGENT_MODEL` (default `claude-sonnet-4-5`; 🔴 confirm current name at
    https://docs.claude.com at build time)
- **[ ] OpenAI** → https://platform.openai.com/api-keys → `OPENAI_API_KEY`
  - model: `OPENAI_MODEL` (default `gpt-4o`)
- **[ ] OpenRouter** → https://openrouter.ai → `OPENROUTER_API_KEY`
  - model: `OPENROUTER_MODEL` (any OpenRouter slug, default `anthropic/claude-3.5-sonnet`)

> One is enough to run. Set `LLM_PROVIDER` to whichever you hold a key for.

## 5. Auth secret
- [ ] Generate JWT secret: `openssl rand -hex 32` → `JWT_SECRET` in `.env`

---

## Reminder
- Copy `.env.example` → `.env`, fill values, and **never** commit `.env`.
- All values above are per-teammate; keep them private.
- If a key is accidentally committed to git, revoke it immediately and regenerate.
