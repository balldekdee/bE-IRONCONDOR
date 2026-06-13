-- ============================================================================
-- 0DTE Break-Even Iron Condor Bot — Supabase Schema
-- ============================================================================
-- รัน SQL นี้ใน Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- ----------------------------------------------------------------------------

-- ── Trades: ทุก Iron Condor ที่เปิด/ปิด ──────────────────────────────────────
create table if not exists trades (
    id                bigint generated always as identity primary key,
    trade_id          text unique not null,
    opened_at         timestamptz not null default now(),
    closed_at         timestamptz,
    underlying_price  numeric,
    call_short_strike numeric,
    call_long_strike  numeric,
    put_short_strike  numeric,
    put_long_strike   numeric,
    call_premium      numeric,
    put_premium       numeric,
    total_premium     numeric,
    stop_loss_value   numeric,
    expiry            date,
    regime_at_entry   int,           -- 0=Risk-on, 1=Transition, 2=Crisis
    regime_probs      jsonb,         -- [p0, p1, p2]
    regime_confidence numeric,
    expected_edge     numeric,       -- edge ที่ filter ประเมินตอนเข้า
    pnl               numeric,
    outcome           text,          -- win | loss | breakeven | stopped | eod
    status            text default 'open'
);

create index if not exists idx_trades_regime  on trades (regime_at_entry);
create index if not exists idx_trades_status  on trades (status);
create index if not exists idx_trades_opened  on trades (opened_at);

-- ── Regime snapshots: posterior ทุก bar (analysis/backtest) ──────────────────
create table if not exists regime_snapshots (
    id                  bigint generated always as identity primary key,
    ts                  timestamptz not null default now(),
    underlying_price    numeric,
    map_regime          int,
    regime_name         text,
    prob_risk_on        numeric,
    prob_transition     numeric,
    prob_crisis         numeric,
    change_point_prob   numeric,
    confidence          numeric,
    latent_vol          numeric,
    expected_run_length numeric
);

create index if not exists idx_regime_ts on regime_snapshots (ts);

-- ── Self-improving posteriors: regime-conditional edge (เรียนรู้ข้ามวัน) ─────
create table if not exists regime_posteriors (
    id         int primary key,        -- singleton row (id=1)
    updated_at timestamptz not null default now(),
    state      jsonb not null           -- serialized SelfImprovingFilter
);

-- ── Model state: ensemble snapshot สำหรับ resume ข้ามวัน ─────────────────────
create table if not exists model_state (
    id         int primary key,        -- singleton row (id=1)
    updated_at timestamptz not null default now(),
    state      jsonb not null           -- serialized RegimeEnsemble (Kalman/BOCPD/HSMM)
);

-- ── Helpful view: regime performance summary ────────────────────────────────
create or replace view regime_performance as
select
    regime_at_entry,
    count(*)                                   as n_trades,
    round(avg(pnl), 2)                         as avg_pnl,
    round(sum(pnl), 2)                         as total_pnl,
    round(100.0 * count(*) filter (where pnl > 0) / nullif(count(*), 0), 1) as win_rate_pct
from trades
where status = 'closed'
group by regime_at_entry
order by regime_at_entry;
