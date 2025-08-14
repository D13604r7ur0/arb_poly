
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitraje Polymarket ↔ Binance Options (one-shot) — v2.7
Cambios:
- Resolver PM por --pm-slug ahora intenta Gamma; si no encuentra, cae a CLOB: /markets?slug=...
  y filtra por la fecha de --expiry (prefijo YYYY-MM-DD).
- Mantiene --pm-condition-id y --pm-token-id como antes.
- Modo up/down, --approx para down sin PUTs≤K.
"""
import argparse, sys, json, math, re, time
from typing import Optional, Tuple, Dict, Any, List
from urllib.request import Request, urlopen
from urllib.parse import urlencode

PM_CLOB = "https://clob.polymarket.com"
PM_GAMMA = "https://gamma-api.polymarket.com"
BIN_EAPI = "https://eapi.binance.com"

DEFAULT_HEADERS = {"Accept": "application/json, text/plain, */*", "User-Agent": "curl/8.6.0"}

def http_get(url: str, params: dict = None, timeout: int = 20) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    try:
        return json.loads(data.decode())
    except Exception:
        return data.decode()

def http_post(url: str, payload: dict, timeout: int = 20) -> Any:
    data = json.dumps(payload).encode()
    headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    try:
        return json.loads(data.decode())
    except Exception:
        return data.decode()

# -------- PM --------
def normalize_outcome(s: str) -> str:
    if s is None: return ""
    t = s.strip().lower().replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("—", "-")
    t = t.replace(" ", "").replace("$", "").replace("k","000")
    return t

def clob_market_by_condition(condition_id: str) -> Optional[Dict[str,Any]]:
    j = http_get(f"{PM_CLOB}/markets/{condition_id}")
    if isinstance(j, dict):
        if isinstance(j.get("market"), dict): return j["market"]
        # algunos devuelven los campos en root
        if j.get("condition_id") or j.get("conditionId") or j.get("id"): return j
    return None

def gamma_markets_by_slug(slug: str) -> List[Dict[str,Any]]:
    res = http_get(f"{PM_GAMMA}/markets", params={"slug": slug}); return res if isinstance(res, list) else []

def gamma_markets_by_condition(condition_id: str) -> List[Dict[str,Any]]:
    res = http_get(f"{PM_GAMMA}/markets", params={"condition_ids": condition_id}); return res if isinstance(res, list) else []

def clob_markets_by_slug(slug: str) -> List[Dict[str,Any]]:
    res = http_get(f"{PM_CLOB}/markets", params={"slug": slug})
    if isinstance(res, dict):
        data = res.get("data")
        if isinstance(data, list): return data
    return []

def ymd_prefix_from_expiry(expiry: str) -> str:
    s = re.sub(r"\D", "", str(expiry))
    if len(s) == 8: ymd = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    elif len(s) == 6: ymd = f"20{s[0:2]}-{s[2:4]}-{s[4:6]}"
    else: raise ValueError("Expiry inválido (usa YYYYMMDD o YYMMDD).")
    return ymd

def resolve_pm_token_id(condition_id: Optional[str], slug: Optional[str], outcome_query: Optional[str], expiry_ymd: Optional[str]) -> Tuple[str, Dict[str,Any]]:
    if not outcome_query: raise ValueError("Debes pasar --pm-outcome (ej: '>118k').")
    oq = normalize_outcome(outcome_query)
    market = None

    if condition_id:
        market = clob_market_by_condition(condition_id)
        if market is None: raise RuntimeError("No pude leer tokens para ese condition_id.")
    else:
        if not slug: raise ValueError("Falta --pm-condition-id o --pm-slug.")
        gl = gamma_markets_by_slug(slug)
        if gl:
            # si hay varios, intenta matchear por fecha
            target = None
            if expiry_ymd:
                for gm in gl:
                    end_iso = gm.get("end_date_iso") or gm.get("endDateISO") or gm.get("end_date") or ""
                    if isinstance(end_iso, str) and end_iso.startswith(expiry_ymd):
                        target = gm; break
            if not target: target = gl[0]
            condition_id = (target.get("condition_ids") or [None])[0] if isinstance(target.get("condition_ids"), list) else target.get("condition_id")
            if not condition_id: raise RuntimeError("Gamma no devolvió condition_id para el slug.")
            market = clob_market_by_condition(condition_id)
            if market is None: raise RuntimeError("No pude leer tokens desde CLOB con el condition_id de Gamma.")
        else:
            # Fallback a CLOB por slug
            cl = clob_markets_by_slug(slug)
            if not cl: raise RuntimeError("No encontré mercado para ese slug (Gamma y CLOB vacíos).")
            target = None
            if expiry_ymd:
                for m in cl:
                    end_iso = m.get("end_date_iso") or m.get("end_date") or ""
                    if isinstance(end_iso, str) and end_iso.startswith(expiry_ymd):
                        target = m; break
            if not target: target = cl[0]
            # target ya viene con tokens? a veces no; entonces volvemos a /markets/<cid>
            cid = target.get("condition_id") or target.get("id") or target.get("conditionId")
            if not cid: raise RuntimeError("CLOB/slug no devolvió condition_id.")
            market = clob_market_by_condition(cid)
            if market is None: raise RuntimeError("No pude leer tokens desde CLOB con el condition_id del slug.")
    tokens = market.get("tokens") or []
    cand = []
    for t in tokens:
        out_norm = normalize_outcome(t.get("outcome",""))
        if oq in out_norm or out_norm in oq: cand.append(t)
    if not cand:
        nums = re.findall(r"\d+", oq)
        if nums:
            needle = nums[-1]
            for t in tokens:
                if needle in normalize_outcome(t.get("outcome","")): cand.append(t)
    if not cand: raise RuntimeError("No encontré token para ese outcome. Outcomes: " + str([t.get("outcome") for t in tokens]))
    tok = cand[0]
    return tok["token_id"], market

def pm_mid_yes(token_id: str) -> float:
    j = http_get(f"{PM_CLOB}/midpoint", params={"token_id": token_id})
    if isinstance(j, dict) and "mid" in j: return float(j["mid"])
    j2 = http_post(f"{PM_CLOB}/midpoints", payload={"params":[{"token_id": token_id}]})
    if isinstance(j2, dict):
        arr = j2.get("midpoints") or j2.get("data") or []
        if arr and isinstance(arr, list):
            mid = arr[0].get("mid") or arr[0].get("price") or arr[0].get("midpoint")
            if mid is not None: return float(mid)
    raise RuntimeError("No pude obtener el midpoint en Polymarket.")

def pm_top_of_book(token_id: str) -> Dict[str, float]:
    """Return best bid/ask and sizes for a PM token."""
    j = http_get(f"{PM_CLOB}/orderbook", params={"token_id": token_id})
    book = {"bid": float("nan"), "bid_sz": 0.0, "ask": float("nan"), "ask_sz": 0.0}
    if isinstance(j, dict):
        bids = j.get("bids") or []
        asks = j.get("asks") or []
        if bids:
            book["bid"] = float(bids[0][0])
            book["bid_sz"] = float(bids[0][1])
        if asks:
            book["ask"] = float(asks[0][0])
            book["ask_sz"] = float(asks[0][1])
    else:
        # fallback via /orders
        bid = http_get(f"{PM_CLOB}/orders", params={"token_id": token_id, "side": "buy", "limit": 1})
        ask = http_get(f"{PM_CLOB}/orders", params={"token_id": token_id, "side": "sell", "limit": 1})
        if isinstance(bid, list) and bid:
            book["bid"] = float(bid[0].get("price", float("nan")))
            book["bid_sz"] = float(bid[0].get("size", 0.0))
        if isinstance(ask, list) and ask:
            book["ask"] = float(ask[0].get("price", float("nan")))
            book["ask_sz"] = float(ask[0].get("size", 0.0))
    return book

# -------- Binance --------
def expiry_to_code(expiry: str) -> str:
    s = re.sub(r"\D", "", str(expiry))
    if len(s) == 8: return s[2:]
    if len(s) == 6: return s
    raise ValueError("Expiry inválido. Usa YYYYMMDD o YYMMDD.")

def load_exchange_info() -> dict:
    return http_get(f"{BIN_EAPI}/eapi/v1/exchangeInfo")

def list_option_symbols(exch: dict, underlying_prefix: str, expiry_code: str, right: str) -> List[Tuple[str,float]]:
    syms = exch.get("optionSymbols") or []
    out = []
    for s in syms:
        sym = s.get("symbol","")
        side = (s.get("side") or s.get("optionType") or "").upper()
        if right.upper().startswith("C") and side and not side.startswith("C"): continue
        if right.upper().startswith("P") and side and not side.startswith("P"): continue
        parts = sym.split("-")
        if len(parts) < 4: continue
        und, exp, strike, r = parts[0], parts[1], parts[2], parts[3]
        if not und.startswith(underlying_prefix.upper()): continue
        if exp != expiry_code: continue
        try: k = float(strike)
        except: continue
        out.append((sym, k))
    return sorted(out, key=lambda x: x[1])

def mark_price(symbol: str) -> Optional[float]:
    j = http_get(f"{BIN_EAPI}/eapi/v1/mark", params={"symbol": symbol})
    if isinstance(j, dict) and "markPrice" in j:
        try: return float(j["markPrice"])
        except: return None
    if isinstance(j, list):
        for it in j:
            if it.get("symbol")==symbol and it.get("markPrice") is not None:
                try: return float(it["markPrice"])
                except: pass
    return None

def depth_mid(symbol: str) -> float:
    j = http_get(f"{BIN_EAPI}/eapi/v1/depth", params={"symbol": symbol, "limit": 20})
    if not isinstance(j, dict) or "bids" not in j or "asks" not in j:
        raise RuntimeError(f"No pude leer orderbook para {symbol}")
    def best(side):
        return float(side[0][0]) if side else float('nan')
    bid = best(j["bids"]); ask = best(j["asks"])
    if math.isnan(bid) and math.isnan(ask): return float('nan')
    if math.isnan(bid): return ask
    if math.isnan(ask): return bid
    return (bid+ask)/2.0

def price_from_symbol(symbol: str) -> float:
    mp = mark_price(symbol)
    return mp if mp is not None else depth_mid(symbol)

def binance_top_of_book(symbol: str, limit: int = 5) -> Dict[str, float]:
    j = http_get(f"{BIN_EAPI}/eapi/v1/depth", params={"symbol": symbol, "limit": limit})
    def best(side):
        return (float(side[0][0]), float(side[0][1])) if side else (float("nan"), 0.0)
    bid, bid_sz = best(j.get("bids", [])) if isinstance(j, dict) else (float("nan"), 0.0)
    ask, ask_sz = best(j.get("asks", [])) if isinstance(j, dict) else (float("nan"), 0.0)
    return {"bid": bid, "bid_sz": bid_sz, "ask": ask, "ask_sz": ask_sz}

def executable_price(side: str, book: Dict[str, List[List[str]]], size_req: float) -> float:
    """Accumulate levels until size_req and return avg executable price."""
    levels = book.get("asks" if side == "buy" else "bids", []) if isinstance(book, dict) else []
    if not levels:
        return float("nan")
    rem = size_req
    total = 0.0
    for px, sz in ((float(p), float(s)) for p, s in levels):
        take = min(rem, sz)
        total += take * px
        rem -= take
        if rem <= 1e-9:
            break
    if rem > 1e-9:
        # not enough liquidity
        return float("nan")
    return total / size_req

# -------- Digital replication --------
def cost_add(pP, pB, pm_bps, opt_bps, slip_bps):
    return abs(pP)*(pm_bps+slip_bps)/10000.0 + abs(pB)*(opt_bps+slip_bps)/10000.0

def compute_up(pm_yes: float, Ck: float, Ck2: float, dk: float, payout: float, pm_bps: float, opt_bps: float, slip_bps: float) -> Dict[str,Any]:
    pB = (Ck - Ck2) / dk * payout
    pP = pm_yes * payout
    cost = cost_add(pP, pB, pm_bps, opt_bps, slip_bps)
    gap = pP - pB; edge = gap - cost
    return {"price_PM": pP, "price_BIN": pB, "gap": gap, "cost": cost, "edge": edge}

def compute_down(pm_yes: float, Pk: float, Pk2: float, dk: float, payout: float, pm_bps: float, opt_bps: float, slip_bps: float) -> Dict[str,Any]:
    pB = (Pk - Pk2) / dk * payout
    pP = pm_yes * payout
    cost = cost_add(pP, pB, pm_bps, opt_bps, slip_bps)
    gap = pP - pB; edge = gap - cost
    return {"price_PM": pP, "price_BIN": pB, "gap": gap, "cost": cost, "edge": edge}

# -------- Scanning helpers --------
def find_markets(query: str, ymd: str) -> List[Dict[str, Any]]:
    """Find PM markets by slug query and date prefix."""
    res: List[Dict[str, Any]] = []
    gl = gamma_markets_by_slug(query) if query else []
    for m in gl:
        end_iso = m.get("end_date_iso") or m.get("endDateISO") or ""
        if not ymd or (isinstance(end_iso, str) and end_iso.startswith(ymd)):
            res.append(m)
    cl = clob_markets_by_slug(query) if query else []
    for m in cl:
        end_iso = m.get("end_date_iso") or m.get("end_date") or ""
        if not ymd or (isinstance(end_iso, str) and end_iso.startswith(ymd)):
            res.append(m)
    out = []
    for m in res:
        tokens = m.get("tokens")
        if not tokens:
            cid = m.get("condition_id") or m.get("id") or m.get("conditionId")
            if cid:
                m2 = clob_market_by_condition(cid)
                if m2:
                    tokens = m2.get("tokens")
        if not tokens:
            continue
        for t in tokens:
            norm = normalize_outcome(t.get("outcome", ""))
            if re.search(r"[<>]\d", norm):
                out.append({"slug": m.get("slug", ""), "token": t})
    return out

def scan_and_rank(query: str, ymd: str, sense: str, dk: int, payout: float, fees: Dict[str,float],
                  min_size: float, use_mark: bool, use_depth: bool) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    exch = load_exchange_info()
    markets = find_markets(query, ymd)
    for m in markets:
        tok = m["token"]
        token_id = tok.get("token_id")
        outcome_norm = normalize_outcome(tok.get("outcome",""))
        nums = re.findall(r"\d+", outcome_norm)
        if not nums:
            continue
        K = float(nums[-1])
        pm_mid = pm_mid_yes(token_id)
        top = pm_top_of_book(token_id)
        if min_size and top.get("ask_sz",0.0) < min_size and top.get("bid_sz",0.0) < min_size:
            continue
        expiry = ymd.replace("-","")
        code = expiry_to_code(expiry)
        if "<" in outcome_norm:
            rt = "P"; sens = "down"
        else:
            rt = "C"; sens = "up"
        if sens != sense:
            continue
        syms = list_option_symbols(exch, "BTC", code, rt)
        if not syms:
            continue
        k1 = None
        if sens == "up":
            for sym, k in syms:
                if k >= K:
                    k1 = (sym, k); break
            if k1 is None: k1 = syms[-1]
            target = k1[1] + dk
            k2 = min(syms, key=lambda x: abs(x[1]-target))
            Ck = mark_price(k1[0]) if use_mark else price_from_symbol(k1[0])
            Ck2 = mark_price(k2[0]) if use_mark else price_from_symbol(k2[0])
            res = compute_up(pm_mid, Ck, Ck2, k2[1]-k1[1], payout, fees.get("pm",0), fees.get("opt",0), fees.get("slip",0))
            ops.append({"pm": {"token_id": token_id, "mid": pm_mid, "top": top, "K": K},
                        "binance": {"K1": k1[1], "K2": k2[1], "sym1": k1[0], "sym2": k2[0]},
                        "results": res})
        else:
            below = [t for t in syms if t[1] <= K]
            if not below:
                continue
            k1 = below[-1]
            smaller = [t for t in syms if t[1] < k1[1]]
            if not smaller:
                continue
            target = k1[1] - dk
            k2 = min(smaller, key=lambda x: abs(x[1]-target))
            Pk = mark_price(k1[0]) if use_mark else price_from_symbol(k1[0])
            Pk2 = mark_price(k2[0]) if use_mark else price_from_symbol(k2[0])
            res = compute_down(pm_mid, Pk, Pk2, k1[1]-k2[1], payout, fees.get("pm",0), fees.get("opt",0), fees.get("slip",0))
            ops.append({"pm": {"token_id": token_id, "mid": pm_mid, "top": top, "K": K},
                        "binance": {"K1": k1[1], "K2": k2[1], "sym1": k1[0], "sym2": k2[0]},
                        "results": res})
    ops.sort(key=lambda x: x["results"]["edge"], reverse=True)
    return ops

def plan_trades(op: Dict[str, Any], budget: float, buy_side: str) -> Dict[str, Any]:
    """Return plan of orders based on budget and preferred buy side."""
    price_pm = op["results"]["price_PM"]
    price_bin = op["results"]["price_BIN"]
    unit = 1.0
    if budget:
        max_leg = max(abs(price_pm), abs(price_bin))
        unit = max(1.0, math.floor(budget / max_leg))
    size_opt = unit / abs(op["binance"]["K2"] - op["binance"]["K1"])
    pm_side = "buy" if buy_side == "pm" else "sell"
    bin_side1 = "sell" if buy_side == "pm" else "buy"
    bin_side2 = "buy" if buy_side == "pm" else "sell"
    plan = {
        "pm": {"token_id": op["pm"]["token_id"], "side": pm_side, "price": op["pm"]["mid"], "size": unit},
        "binance": [
            {"symbol": op["binance"]["sym1"], "side": bin_side1, "price": price_bin, "quantity": size_opt},
            {"symbol": op["binance"]["sym2"], "side": bin_side2, "price": price_bin, "quantity": size_opt}
        ]
    }
    return plan

def monitor_loop(args):
    while True:
        ops = scan_and_rank(args.pm_slug, ymd_prefix_from_expiry(args.expiry), args.sense, args.dk, args.payout,
                            {"pm": args.pm_bps, "opt": args.opt_bps, "slip": args.slip_bps},
                            args.min_size, args.use_mark, args.use_depth)
        best = max(ops, key=lambda o: o["results"]["edge"], default=None)
        if best and best["results"]["edge"] >= args.min_edge:
            plan = plan_trades(best, args.budget, args.buy_side)
            print(json.dumps({"opportunity": best, "plan": plan}, indent=2))
            if args.dry_run:
                return
            if args.confirm and input("¿Ejecutar plan? [y/N]: ").lower() != "y":
                return
            return
        time.sleep(args.interval)

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-token-id")
    ap.add_argument("--pm-condition-id")
    ap.add_argument("--pm-slug")
    ap.add_argument("--pm-outcome")
    ap.add_argument("--underlying", default="BTC")
    ap.add_argument("--expiry", required=True, help="YYYYMMDD o YYMMDD (usado también para elegir el market por fecha)")
    ap.add_argument("--strike", type=float, help="K del evento PM")
    ap.add_argument("--dk", type=int, default=100, help="Δ deseado")
    ap.add_argument("--payout", type=float, default=1000.0)
    ap.add_argument("--pm-bps", type=float, default=0.0)
    ap.add_argument("--opt-bps", type=float, default=0.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--sense", choices=["up","down"], default="up", help="Evento PM: 'up' = S≥K, 'down' = S<K")
    ap.add_argument("--approx", action="store_true", help="Permite aproximar 'down' con 1 - digital_up(K*) si faltan PUT≤K")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--min-size", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--use-mark", action="store_true")
    ap.add_argument("--use-depth", action="store_true")
    ap.add_argument("--buy-side", choices=["pm","binance"], default="pm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--max-shortfall", type=float, default=0.1)
    ap.add_argument("--out", choices=["json","csv","pretty"], default="json")
    args = ap.parse_args()

    if args.scan or args.monitor:
        ymd = ymd_prefix_from_expiry(args.expiry)
        if args.monitor:
            monitor_loop(args)
            return
        ops = scan_and_rank(args.pm_slug, ymd, args.sense, args.dk, args.payout,
                            {"pm": args.pm_bps, "opt": args.opt_bps, "slip": args.slip_bps},
                            args.min_size, args.use_mark, args.use_depth)
        if args.out == "json":
            print(json.dumps(ops, indent=2))
        elif args.out == "csv":
            print("slug,token_id,K1,K2,price_PM,price_BIN,edge")
            for op in ops:
                pm = op["pm"]; bn = op["binance"]; r = op["results"]
                print(f"{args.pm_slug},{pm['token_id']},{bn['K1']},{bn['K2']},{r['price_PM']},{r['price_BIN']},{r['edge']}")
        else:
            print(json.dumps(ops, indent=2))
        return

    if args.strike is None:
        raise RuntimeError("Debes pasar --strike para modo one-shot")

    ymd = ymd_prefix_from_expiry(args.expiry)

    # PM mid
    tok = args.pm_token_id
    if not tok:
        tok, _ = resolve_pm_token_id(args.pm_condition_id, args.pm_slug, args.pm_outcome, ymd)
    pm_mid = pm_mid_yes(tok)

    # Binance
    code = re.sub(r"\D", "", str(args.expiry))[2:] if len(re.sub(r"\D","",str(args.expiry)))==8 else re.sub(r"\D","",str(args.expiry))
    exch = load_exchange_info()

    if args.sense == "up":
        calls = list_option_symbols(exch, args.underlying, code, "C")
        if not calls: raise RuntimeError(f"No hay CALLs para expiry {code}")
        k1 = None
        for sym,k in calls:
            if k >= args.strike: k1=(sym,k); break
        if k1 is None: k1 = calls[-1]
        target = k1[1] + args.dk
        k2 = min(calls, key=lambda x: abs(x[1]-target))
        if k2[1] <= k1[1]:
            bigger = [t for t in calls if t[1] > k1[1]]
            if not bigger: raise RuntimeError("No hay strike más alto para formar spread")
            k2 = bigger[0]
        Ck  = price_from_symbol(k1[0])
        Ck2 = price_from_symbol(k2[0])
        dk  = (k2[1]-k1[1])
        res = compute_up(pm_mid, Ck, Ck2, dk, args.payout, args.pm_bps, args.opt_bps, args.slip_bps)
        out = {"mode":"up","pm":{"mid_yes":pm_mid},"binance":{"K":k1[1],"K2":k2[1],"C(K)":Ck,"C(K2)":Ck2,"Δ":dk},"results":res}
    else:
        puts = list_option_symbols(exch, args.underlying, code, "P")
        below_eq = [t for t in puts if t[1] <= args.strike]
        if not below_eq:
            if not args.approx:
                raise RuntimeError("No hay strikes de PUT ≤ K para replicar 1{S<K} en esa expiración. Usa --approx para complemento con CALLs.")
            calls = list_option_symbols(exch, args.underlying, code, "C")
            if not calls: raise RuntimeError("No hay CALLs para usar complemento.")
            k1 = None
            for sym,k in calls:
                if k >= args.strike: k1=(sym,k); break
            if k1 is None: k1 = calls[-1]
            target = k1[1] + args.dk
            k2 = min(calls, key=lambda x: abs(x[1]-target))
            if k2[1] <= k1[1]:
                bigger = [t for t in calls if t[1] > k1[1]]
                if not bigger: raise RuntimeError("No hay strike más alto para formar spread")
                k2 = bigger[0]
            Ck  = price_from_symbol(k1[0])
            Ck2 = price_from_symbol(k2[0])
            dk  = (k2[1]-k1[1])
            digital_up = (Ck - Ck2) / dk
            pB = (1.0 - digital_up) * args.payout
            pP = pm_mid * args.payout
            cost = abs(pP)*(args.pm_bps+args.slip_bps)/10000.0 + abs(pB)*(args.opt_bps+args.slip_bps)/10000.0
            gap = pP - pB; edge = gap - cost
            out = {"mode":"down-approx","note":"Usando 1 - digital_up(K*). Basis risk por K*≥K y Δ≠deseado.",
                   "pm":{"mid_yes":pm_mid},
                   "binance":{"K*":k1[1],"K2":k2[1],"C(K*)":Ck,"C(K2)":Ck2,"Δ":dk},
                   "results":{"price_PM":pP,"price_BIN":pB,"gap":gap,"cost":cost,"edge":edge}}
        else:
            k1 = below_eq[-1]
            below = [t for t in puts if t[1] < k1[1]]
            if not below:
                if not args.approx:
                    raise RuntimeError("No hay strike inferior para formar spread de PUTs.")
                else:
                    raise RuntimeError("No hay strike inferior de PUT ni complemento posible.")
            target = k1[1] - args.dk
            k2 = min(below, key=lambda x: abs(x[1]-target))
            Pk  = price_from_symbol(k1[0])
            Pk2 = price_from_symbol(k2[0])
            dk  = (k1[1]-k2[1])
            res = compute_down(pm_mid, Pk, Pk2, dk, args.payout, args.pm_bps, args.opt_bps, args.slip_bps)
            out = {"mode":"down","pm":{"mid_yes":pm_mid},"binance":{"K":k1[1],"K2":k2[1],"P(K)":Pk,"P(K2)":Pk2,"Δ":dk},"results":res}
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
