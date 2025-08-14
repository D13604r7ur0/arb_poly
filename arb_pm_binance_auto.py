#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitraje Polymarket ↔ Binance Options — v2.9 (modo MENÚ)
- Te deja seleccionar interactivamente:
  1) Activo (BTC / ETH)
  2) Mercado de Polymarket (lista por fecha/pregunta) y outcome (>xxxk o <xxxk)
  3) Expiración de Binance (sugerida por la fecha del market) y Δ deseado
- Luego calcula precios, edge y muestra tabla de escenarios de payoff.
"""
import argparse, json, math, re, sys, time
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

# ---------- PM helpers ----------
def normalize_outcome(s: str) -> str:
    if s is None: return ""
    t = s.strip().lower().replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("—", "-")
    t = t.replace(" ", "").replace("$", "").replace("k","000")
    return t

def clob_markets_by_slug(slug: str) -> List[Dict[str,Any]]:
    res = http_get(f"{PM_CLOB}/markets", params={"slug": slug, "limit": 500})
    if isinstance(res, dict):
        data = res.get("data")
        if isinstance(data, list): return data
    return []

def clob_market_by_condition(condition_id: str) -> Optional[Dict[str,Any]]:
    j = http_get(f"{PM_CLOB}/markets/{condition_id}")
    if isinstance(j, dict):
        if isinstance(j.get("market"), dict): return j["market"]
        if j.get("condition_id") or j.get("conditionId") or j.get("id"): return j
    return None

def gamma_markets_by_slug(slug: str):
    """Devuelve lista de markets desde Gamma por slug (o [] si no hay)."""
    try:
        res = http_get(f"{PM_GAMMA}/markets", params={"slug": slug, "limit": 500})
        if isinstance(res, list):
            return res
        if isinstance(res, dict):
            return res.get("data") or res.get("markets") or []
    except Exception:
        pass
    return []

def gamma_markets_by_condition(condition_id: str) -> List[Dict[str,Any]]:
    res = http_get(f"{PM_GAMMA}/markets", params={"condition_ids": condition_id})
    return res if isinstance(res, list) else []

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
    """Mejor bid/ask y tamaños para un token PM."""
    j = http_get(f"{PM_CLOB}/orderbook", params={"token_id": token_id})
    book = {"bid": float("nan"), "bid_sz": 0.0, "ask": float("nan"), "ask_sz": 0.0}
    if isinstance(j, dict):
        bids = j.get("bids") or []
        asks = j.get("asks") or []
        if bids:
            book["bid"] = float(bids[0][0]); book["bid_sz"] = float(bids[0][1])
        if asks:
            book["ask"] = float(asks[0][0]); book["ask_sz"] = float(asks[0][1])
    else:
        bid = http_get(f"{PM_CLOB}/orders", params={"token_id": token_id, "side": "buy", "limit": 1})
        ask = http_get(f"{PM_CLOB}/orders", params={"token_id": token_id, "side": "sell", "limit": 1})
        if isinstance(bid, list) and bid:
            book["bid"] = float(bid[0].get("price", float("nan")))
            book["bid_sz"] = float(bid[0].get("size", 0.0))
        if isinstance(ask, list) and ask:
            book["ask"] = float(ask[0].get("price", float("nan")))
            book["ask_sz"] = float(ask[0].get("size", 0.0))
    return book

def ymd_prefix_from_expiry(expiry: str) -> str:
    s = re.sub(r"\D", "", str(expiry))
    if len(s) == 8: return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    if len(s) == 6: return f"20{s[0:2]}-{s[2:4]}-{s[4:6]}"
    raise ValueError("Expiry inválido (usa YYYYMMDD o YYMMDD).")

def resolve_pm_token_id(condition_id: Optional[str], slug: Optional[str], outcome_query: Optional[str], expiry_ymd: Optional[str]) -> Tuple[str, Dict[str,Any]]:
    if not outcome_query:
        raise ValueError("Debes pasar --pm-outcome (ej: '>118k').")
    oq = normalize_outcome(outcome_query)
    market = None
    if condition_id:
        market = clob_market_by_condition(condition_id)
        if market is None:
            raise RuntimeError("No pude leer tokens para ese condition_id.")
    else:
        if not slug:
            raise ValueError("Falta --pm-condition-id o --pm-slug.")
        gl = gamma_markets_by_slug(slug)
        if gl:
            target = None
            if expiry_ymd:
                for gm in gl:
                    end_iso = gm.get("end_date_iso") or gm.get("endDateISO") or gm.get("end_date") or ""
                    if isinstance(end_iso, str) and end_iso.startswith(expiry_ymd):
                        target = gm; break
            if not target:
                target = gl[0]
            condition_id = (target.get("condition_ids") or [None])[0] if isinstance(target.get("condition_ids"), list) else target.get("condition_id")
            if not condition_id:
                raise RuntimeError("Gamma no devolvió condition_id para el slug.")
            market = clob_market_by_condition(condition_id)
            if market is None:
                raise RuntimeError("No pude leer tokens desde CLOB con el condition_id de Gamma.")
        else:
            cl = clob_markets_by_slug(slug)
            if not cl:
                raise RuntimeError("No encontré mercado para ese slug (Gamma y CLOB vacíos).")
            target = None
            if expiry_ymd:
                for m in cl:
                    end_iso = m.get("end_date_iso") or m.get("end_date") or ""
                    if isinstance(end_iso, str) and end_iso.startswith(expiry_ymd):
                        target = m; break
            if not target:
                target = cl[0]
            cid = target.get("condition_id") or target.get("id") or target.get("conditionId")
            if not cid:
                raise RuntimeError("CLOB/slug no devolvió condition_id.")
            market = clob_market_by_condition(cid)
            if market is None:
                raise RuntimeError("No pude leer tokens desde CLOB con el condition_id del slug.")
    tokens = market.get("tokens") or []
    cand = []
    for t in tokens:
        out_norm = normalize_outcome(t.get("outcome",""))
        if oq in out_norm or out_norm in oq:
            cand.append(t)
    if not cand:
        nums = re.findall(r"\d+", oq)
        if nums:
            needle = nums[-1]
            for t in tokens:
                if needle in normalize_outcome(t.get("outcome","")):
                    cand.append(t)
    if not cand:
        raise RuntimeError("No encontré token para ese outcome. Outcomes: " + str([t.get("outcome") for t in tokens]))
    tok = cand[0]
    return tok["token_id"], market

# ---------- Binance helpers ----------
def expiry_to_code(expiry_ymd: str) -> str:
    s = re.sub(r"\D", "", str(expiry_ymd))
    if len(s) == 8: return s[2:]
    if len(s) == 6: return s
    raise ValueError("Expiry inválido. Usa YYYYMMDD o YYMMDD.")

def load_exchange_info() -> dict:
    return http_get(f"{BIN_EAPI}/eapi/v1/exchangeInfo")

def list_option_symbols(exch: dict, underlying_prefix: str) -> List[str]:
    syms = exch.get("optionSymbols") or []
    out = []
    for s in syms:
        sym = s.get("symbol","")
        if sym.startswith(underlying_prefix.upper()+"-"): out.append(sym)
    return sorted(out)

def unique_expiries_for(exch: dict, underlying_prefix: str) -> List[str]:
    # devuelve códigos YYMMDD ordenados
    syms = list_option_symbols(exch, underlying_prefix)
    exps = sorted({sym.split("-")[1] for sym in syms if len(sym.split("-"))>=4})
    return exps

def strikes_for(exch: dict, underlying_prefix: str, exp_code: str, right: str) -> List[Tuple[str,float]]:
    syms = exch.get("optionSymbols") or []
    out = []
    for s in syms:
        sym = s.get("symbol","")
        if not sym.startswith(underlying_prefix.upper()+"-"): continue
        parts = sym.split("-")
        if len(parts)<4: continue
        und, exp, strike, r = parts[0], parts[1], parts[2], parts[3].upper()
        if exp != exp_code: continue
        if right.upper().startswith("C") and r!="C": continue
        if right.upper().startswith("P") and r!="P": continue
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
    def best(side): return float(side[0][0]) if side else float('nan')
    bid = best(j["bids"]); ask = best(j["asks"])
    if math.isnan(bid) and math.isnan(ask): return float('nan')
    if math.isnan(bid): return ask
    if math.isnan(ask): return bid
    return (bid+ask)/2.0

def price_from_symbol(symbol: str, use_mark=True) -> float:
    if use_mark:
        mp = mark_price(symbol)
        if mp is not None: return mp
    return depth_mid(symbol)

def binance_top_of_book(symbol: str, limit: int = 5) -> Dict[str, float]:
    j = http_get(f"{BIN_EAPI}/eapi/v1/depth", params={"symbol": symbol, "limit": limit})
    def best(side):
        return (float(side[0][0]), float(side[0][1])) if side else (float("nan"), 0.0)
    bid, bid_sz = best(j.get("bids", [])) if isinstance(j, dict) else (float("nan"), 0.0)
    ask, ask_sz = best(j.get("asks", [])) if isinstance(j, dict) else (float("nan"), 0.0)
    return {"bid": bid, "bid_sz": bid_sz, "ask": ask, "ask_sz": ask_sz}

def executable_price(side: str, book: Dict[str, List[List[str]]], size_req: float) -> float:
    """Precio medio ejecutable acumulando niveles hasta size_req."""
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
        return float("nan")
    return total / size_req

# ---------- Digitals y escenarios ----------
def digital_up_payoff(ST: float, K1: float, K2: float, payout: float) -> float:
    d = max(1e-9, K2 - K1)
    x = max(0.0, min((ST - K1) / d, 1.0))
    return payout * x

def PM_up_payoff(ST: float, K: float, payout: float) -> float:
    return payout * (1.0 if ST > K else 0.0)

def digital_down_payoff(ST: float, K1: float, K2: float, payout: float) -> float:
    d = max(1e-9, K1 - K2)
    x = max(0.0, min((K1 - ST) / d, 1.0))
    return payout * x

def scenarios_up(K: float, K1: float, K2: float, payout: float) -> List[Dict[str,float]]:
    d = K2 - K1
    pts = [K- d, K-1, K, K+1, K+d/4, K+d/2, K+d-1, K+d, K+1.5*d, K+2*d]
    out=[]
    for ST in pts:
        out.append({
            "S_T": round(ST,2),
            "PM_bin": PM_up_payoff(ST,K,payout),
            "Spread": digital_up_payoff(ST,K1,K2,payout),
            "Error": PM_up_payoff(ST,K,payout) - digital_up_payoff(ST,K1,K2,payout)
        })
    return out

def scenarios_down(K: float, K1: float, K2: float, payout: float) -> List[Dict[str,float]]:
    d = K1 - K2
    pts = [K-2*d, K-1.5*d, K-d, K-1, K, K+1, K+d]
    out=[]
    for ST in pts:
        pm = (1.0 if ST < K else 0.0)*payout
        sp = digital_down_payoff(ST,K1,K2,payout)
        out.append({"S_T": round(ST,2), "PM_bin": pm, "Spread": sp, "Error": pm-sp})
    return out

# ---------- Cálculos ----------
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

# ---------- Scan & monitor ----------
def find_markets(query: str, ymd: str) -> List[Dict[str, Any]]:
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
    out: List[Dict[str, Any]] = []
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

def scan_and_rank(query: str, ymd: str, sense: str, dk: int, payout: float, fees: Dict[str,float], min_size: float) -> List[Dict[str, Any]]:
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
        syms = strikes_for(exch, "BTC", code, rt)
        if not syms:
            continue
        if sens == "up":
            k1 = None
            for sym,k in syms:
                if k >= K:
                    k1 = (sym,k); break
            if k1 is None:
                k1 = syms[-1]
            target = k1[1] + dk
            k2 = min(syms, key=lambda x: abs(x[1]-target))
            Ck = price_from_symbol(k1[0])
            Ck2 = price_from_symbol(k2[0])
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
            Pk = price_from_symbol(k1[0])
            Pk2 = price_from_symbol(k2[0])
            res = compute_down(pm_mid, Pk, Pk2, k1[1]-k2[1], payout, fees.get("pm",0), fees.get("opt",0), fees.get("slip",0))
            ops.append({"pm": {"token_id": token_id, "mid": pm_mid, "top": top, "K": K},
                        "binance": {"K1": k1[1], "K2": k2[1], "sym1": k1[0], "sym2": k2[0]},
                        "results": res})
    ops.sort(key=lambda x: x["results"]["edge"], reverse=True)
    return ops

def plan_trades(op: Dict[str, Any], budget: float, buy_side: str) -> Dict[str, Any]:
    """Plan de órdenes según presupuesto y lado comprador."""
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
                            args.min_size)
        best = max(ops, key=lambda o: o["results"]["edge"], default=None)
        if best and best["results"]["edge"] >= args.min_edge:
            plan = plan_trades(best, args.budget, args.buy_side)
            print(json.dumps({"opportunity": best, "plan": plan}, indent=2, ensure_ascii=False))
            if args.dry_run:
                return
            if args.confirm and input("¿Ejecutar plan? [y/N]: ").lower() != "y":
                return
            return
        time.sleep(args.interval)

# ---------- Menú ----------
def menu_select_asset() -> str:
    print("\nActivos disponibles:\n  [1] BTC\n  [2] ETH")
    while True:
        x = input("Elige activo [1/2]: ").strip()
        if x=="1": return "BTC"
        if x=="2": return "ETH"
        print("Opción inválida.")

def fetch_pm_candidates(asset: str) -> List[Dict[str,Any]]:
    # 1) slugs por activo
    slug_map = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth", "ether"]
    }
    slugs = slug_map[asset]

    def is_asset_in_question(q: str) -> bool:
        ql = (q or "").lower()
        if asset == "BTC":
            return re.search(r"\b(bitcoin|btc)\b", ql) is not None
        else:
            return re.search(r"\b(ethereum|ether|eth)\b", ql) is not None

    # 2) traer candidatos desde CLOB y Gamma
    raw: List[Dict[str,Any]] = []
    for s in slugs:
        cl = clob_markets_by_slug(s) or []
        gm = gamma_markets_by_slug(s) or []
        raw.extend(cl)
        raw.extend(gm)

    seen = set()
    out: List[Dict[str,Any]] = []

    for m in raw:
        cid = m.get("condition_id") or m.get("id") or m.get("conditionId")
        if not cid or cid in seen:
            continue

        question = m.get("question") or ""
        if not is_asset_in_question(question):
            continue

        # 3) tokens: si faltan, pedirlos por condition_id
        tokens = m.get("tokens") or []
        if not tokens:
            m2 = clob_market_by_condition(cid)
            if isinstance(m2, dict):
                tokens = m2.get("tokens") or []

        if not tokens:
            continue

        # 4) outcomes tipo umbral (> / < con número)
        outs = []
        for t in tokens:
            o = (t.get("outcome") or "").lower()
            if re.search(r'[<>]=?\s*\$?\s*\d+(k|,?\d{3})?', o):
                outs.append(t.get("outcome") or "")

        if not outs:
            # mercados Yes/No puros no sirven para replicar digital con spreads
            continue

        end_iso = (m.get("end_date_iso") or m.get("end_date") or
                   m.get("endDateISO") or m.get("endDate") or "")

        seen.add(cid)
        out.append({
            "cid": cid,
            "question": m.get("question") or question,
            "end_date_iso": end_iso,
            "market_slug": m.get("market_slug") or m.get("slug"),
            "outcomes": outs
        })

    # 5) orden temporal
    out.sort(key=lambda r: r.get("end_date_iso") or "")
    return out

def menu_select_pm_market(asset: str, cands: List[Dict[str,Any]]) -> Tuple[str, Dict[str,Any]]:
    if not cands:
        raise RuntimeError("No encontré markets de Polymarket con thresholds para ese activo.")
    print("\nMercados Polymarket (elige uno):")
    for i, r in enumerate(cands, 1):
        eid = (r.get("end_date_iso") or "")[:10]
        outs = ", ".join(r.get("outcomes", [])[:6])
        print(f"[{i:02d}] {eid}  {r.get('question')}  | outcomes: {outs}")
    while True:
        i = input("Nº de mercado: ").strip()
        if i.isdigit() and 1 <= int(i) <= len(cands):
            sel = cands[int(i)-1]
            # refresca tokens del cid para tener token_id
            m = clob_market_by_condition(sel["cid"])
            if not m:
                raise RuntimeError("CLOB no devolvió el mercado elegido.")
            return sel["cid"], m
        print("Índice inválido.")

def menu_select_outcome(market: Dict[str,Any]) -> Tuple[str, float, str]:
    tokens = market.get("tokens") or []
    outs = [
        (t.get("outcome", ""), t.get("token_id"))
        for t in tokens
        if re.search(r'[<>]=?\s*\$?\d+(k|,?\d{3})?', t.get("outcome", "").lower())
    ]
    if not outs:
        raise RuntimeError("Ese mercado no tiene outcomes tipo >/< umbral.")
    print("\nOutcomes disponibles:")
    for i,(o,tid) in enumerate(outs,1):
        print(f"[{i:02d}] {o}")
    while True:
        i = input("Elige outcome: ").strip()
        if i.isdigit() and 1<=int(i)<=len(outs):
            outcome, tok = outs[int(i)-1]
            on = normalize_outcome(outcome)
            # extrae K (en enteros)
            nums = re.findall(r'\d+', on)
            if not nums:
                raise RuntimeError("No pude extraer strike del outcome.")
            K = float(nums[-1])
            sense = "up" if on.startswith(">") else "down"
            return tok, K, sense
        print("Índice inválido.")

def menu_select_binance(asset: str, end_date_iso: str, K_guess: float, sense: str) -> Tuple[str,float,str,float,float]:
    exch = load_exchange_info()
    exps = unique_expiries_for(exch, asset)
    if not exps: raise RuntimeError("No hay expiraciones de opciones en Binance para ese activo.")
    # sugerir por fecha de PM
    ymd = (end_date_iso or "")[:10].replace("-","")
    sugg = expiry_to_code(ymd) if ymd else None
    print("\nExpiraciones Binance (YYMMDD):")
    idx_sugg = None
    for i, e in enumerate(exps, 1):
        mark = ""
        if sugg and e == sugg:
            mark = "  <-- sugerida por fecha PM"
            idx_sugg = i
        print(f"[{i:02d}] {e}{mark}")
    x = input(f"Elige expiración [{idx_sugg if idx_sugg else 1}]: ").strip()
    if not x:
        x = idx_sugg or 1
    x = int(x)
    if not (1 <= x <= len(exps)): raise RuntimeError("Expiración inválida.")
    exp = exps[x-1]

    # Δ por defecto
    dk_default = 2000 if asset=="BTC" else 50
    dk_str = input(f"Δ deseado (default {dk_default}): ").strip()
    dk = float(dk_str) if dk_str else dk_default

    # elegimos strikes y símbolos
    if sense=="up":
        calls = strikes_for(exch, asset, exp, "C")
        if not calls: raise RuntimeError("No hay CALLs en esa expiración.")
        k1 = None
        for sym,k in calls:
            if k >= K_guess: k1=(sym,k); break
        if k1 is None: k1 = calls[-1]
        target = k1[1] + dk
        k2 = min(calls, key=lambda x: abs(x[1]-target))
        if k2[1] <= k1[1]:
            bigger = [t for t in calls if t[1] > k1[1]]
            if not bigger: raise RuntimeError("No hay strike superior para formar spread.")
            k2 = bigger[0]
        use_mark = True
        return exp, dk, k1[0], k2[0], use_mark
    else:
        puts = strikes_for(exch, asset, exp, "P")
        below_eq = [t for t in puts if t[1] <= K_guess]
        if below_eq:
            k1 = below_eq[-1]
            below = [t for t in puts if t[1] < k1[1]]
            if not below: raise RuntimeError("No hay strike inferior para formar put-spread.")
            target = k1[1] - dk
            k2 = min(below, key=lambda x: abs(x[1]-target))
            use_mark = True
            return exp, dk, k1[0], k2[0], use_mark
        else:
            print("No hay PUTs ≤ K; se usará aproximación con CALL-spread y complemento (más basis risk).")
            calls = strikes_for(exch, asset, exp, "C")
            if not calls: raise RuntimeError("No hay CALLs para complemento.")
            k1 = None
            for sym,k in calls:
                if k >= K_guess: k1=(sym,k); break
            if k1 is None: k1 = calls[-1]
            target = k1[1] + dk
            k2 = min(calls, key=lambda x: abs(x[1]-target))
            if k2[1] <= k1[1]:
                bigger=[t for t in calls if t[1] > k1[1]]
                if not bigger: raise RuntimeError("No hay strike superior.")
                k2 = bigger[0]
            use_mark = True
            # devolvemos CALLs; el caller sabrá que es approx down
            return exp, dk, k1[0], k2[0], use_mark

def run_menu(debug: bool = False):
    print("\n=== MODO MENÚ ===")
    asset = menu_select_asset()
    cands = fetch_pm_candidates(asset)
    if debug:
        print(f"\nDEBUG: candidates={len(cands)}")
    cid, market = menu_select_pm_market(asset, cands)
    token_id, K_event, sense = menu_select_outcome(market)
    end_date_iso = market.get("end_date_iso") or market.get("end_date") or ""
    pm_mid = pm_mid_yes(token_id)
    print(f"\nPM elegido:\n  token_id: {token_id}\n  pregunta: {market.get('question')}\n  fecha: {end_date_iso}\n  outcome K={K_event:.0f} ({'>' if sense=='up' else '<'}), mid={pm_mid:.4f}")

    exp, dk, sym1, sym2, use_mark = menu_select_binance(asset, end_date_iso, K_event, sense)
    print(f"\nBinance:\n  expiry={exp}  Δ={dk}\n  patas: {sym1}  |  {sym2}")

    payout_str = input("Payout base (default 1): ").strip()
    payout = float(payout_str) if payout_str else 1.0
    pm_bps = float(input("PM bps (default 0): ") or 0)
    opt_bps = float(input("Options bps (default 0): ") or 0)
    slip_bps = float(input("Slippage bps (default 0): ") or 0)

    # Precios
    p1 = price_from_symbol(sym1, use_mark=use_mark)
    p2 = price_from_symbol(sym2, use_mark=use_mark)
    # K1/K2
    k1 = float(sym1.split("-")[2]); k2 = float(sym2.split("-")[2])
    dk_real = abs(k2 - k1)

    if sense=="up":
        res = compute_up(pm_mid, p1, p2, dk_real, payout, pm_bps, opt_bps, slip_bps)
        scen = scenarios_up(K_event, k1, k2, payout)
        note = ""
    else:
        # intenta PUT-spread; si trajimos CALLs, usamos complemento aprox
        is_calls = sym1.endswith("-C") and sym2.endswith("-C")
        if not is_calls:
            res = compute_down(pm_mid, p1, p2, dk_real, payout, pm_bps, opt_bps, slip_bps)
            scen = scenarios_down(K_event, k1, k2, payout)
            note = ""
        else:
            digital_up = (p1 - p2) / dk_real
            pB = (1.0 - digital_up) * payout
            pP = pm_mid * payout
            cost = cost_add(pP, pB, pm_bps, opt_bps, slip_bps)
            gap = pP - pB; edge = gap - cost
            res = {"price_PM": pP, "price_BIN": pB, "gap": gap, "cost": cost, "edge": edge}
            scen = scenarios_down(K_event, k1, k2, payout)
            note = "Aprox DOWN via 1 - digital_up(K1,K2) con CALLs (más basis risk)."

    out = {
        "input":{
            "asset": asset,
            "pm":{"token_id": token_id, "end_date_iso": end_date_iso, "K": K_event, "sense": sense, "mid_yes": pm_mid},
            "binance":{"expiry": exp, "K1": k1, "K2": k2, "sym1": sym1, "sym2": sym2, "use_mark": use_mark, "Δ_real": dk_real},
            "params":{"payout": payout, "fees_bps":{"pm": pm_bps, "opt": opt_bps, "slip": slip_bps}}
        },
        "results": res,
        "note": note,
        "scenarios": scen
    }
    print("\n" + json.dumps(out, indent=2, ensure_ascii=False))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", action="store_true", help="Lanza el menú interactivo (default)")
    ap.add_argument("--pm-token-id")
    ap.add_argument("--pm-condition-id")
    ap.add_argument("--pm-slug")
    ap.add_argument("--pm-outcome")
    ap.add_argument("--underlying", default="BTC")
    ap.add_argument("--expiry", help="YYYYMMDD o YYMMDD")
    ap.add_argument("--strike", type=float)
    ap.add_argument("--dk", type=int, default=100)
    ap.add_argument("--payout", type=float, default=1.0)
    ap.add_argument("--pm-bps", type=float, default=0.0)
    ap.add_argument("--opt-bps", type=float, default=0.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--sense", choices=["up","down"])
    ap.add_argument("--approx", action="store_true", help="Permite aproximar 'down' con 1 - digital_up(K*) si faltan PUT≤K")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--min-size", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--buy-side", choices=["pm","binance"], default="pm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", choices=["json","csv","pretty"], default="json")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # modo menú por defecto
    if args.menu or not (args.pm_token_id or args.pm_condition_id or args.pm_slug):
        return run_menu(debug=args.debug)

    if args.scan or args.monitor:
        if not (args.pm_slug and args.expiry and args.sense):
            raise SystemExit("Para scan/monitor necesitas --pm-slug, --expiry y --sense.")
        if args.monitor:
            monitor_loop(args)
            return
        ymd = ymd_prefix_from_expiry(args.expiry)
        ops = scan_and_rank(args.pm_slug, ymd, args.sense, args.dk, args.payout,
                            {"pm": args.pm_bps, "opt": args.opt_bps, "slip": args.slip_bps},
                            args.min_size)
        if args.out == "csv":
            print("token_id,K1,K2,price_PM,price_BIN,edge")
            for op in ops:
                pm = op["pm"]; bn = op["binance"]; r = op["results"]
                print(f"{pm['token_id']},{bn['K1']},{bn['K2']},{r['price_PM']},{r['price_BIN']},{r['edge']}")
        else:
            txt = json.dumps(ops, indent=2 if args.out=="pretty" else None, ensure_ascii=False)
            print(txt)
        return

    if (not args.scan and not args.monitor):
        if not (args.expiry and args.strike and args.sense):
            raise SystemExit("Debes pasar --expiry, --strike y --sense cuando no usas --menu.")

    ymd = ymd_prefix_from_expiry(args.expiry)
    tok = args.pm_token_id
    market = None
    if not tok:
        tok, market = resolve_pm_token_id(args.pm_condition_id, args.pm_slug, args.pm_outcome, ymd)
    else:
        market = clob_market_by_condition(args.pm_condition_id) if args.pm_condition_id else None
    pm_mid = pm_mid_yes(tok)

    exch = load_exchange_info()
    code = expiry_to_code(args.expiry)
    if args.sense == "up":
        calls = strikes_for(exch, args.underlying, code, "C")
        if not calls:
            raise RuntimeError("No hay CALLs para esa expiración.")
        k1 = None
        for sym, k in calls:
            if k >= args.strike:
                k1 = (sym, k); break
        if k1 is None:
            k1 = calls[-1]
        target = k1[1] + args.dk
        k2 = min(calls, key=lambda x: abs(x[1]-target))
        if k2[1] <= k1[1]:
            bigger = [t for t in calls if t[1] > k1[1]]
            if not bigger:
                raise RuntimeError("No hay strike superior para formar spread.")
            k2 = bigger[0]
        Ck = price_from_symbol(k1[0])
        Ck2 = price_from_symbol(k2[0])
        dk = (k2[1] - k1[1])
        res = compute_up(pm_mid, Ck, Ck2, dk, args.payout, args.pm_bps, args.opt_bps, args.slip_bps)
        scen = scenarios_up(args.strike, k1[1], k2[1], args.payout)
        note = ""
    else:
        puts = strikes_for(exch, args.underlying, code, "P")
        below_eq = [t for t in puts if t[1] <= args.strike]
        if below_eq:
            k1 = below_eq[-1]
            below = [t for t in puts if t[1] < k1[1]]
            if not below:
                raise RuntimeError("No hay strike inferior para formar spread de PUTs.")
            target = k1[1] - args.dk
            k2 = min(below, key=lambda x: abs(x[1]-target))
            Pk = price_from_symbol(k1[0])
            Pk2 = price_from_symbol(k2[0])
            dk = (k1[1] - k2[1])
            res = compute_down(pm_mid, Pk, Pk2, dk, args.payout, args.pm_bps, args.opt_bps, args.slip_bps)
            scen = scenarios_down(args.strike, k1[1], k2[1], args.payout)
            note = ""
        else:
            if not args.approx:
                raise RuntimeError("No hay PUTs ≤ K; usa --approx para complementar con CALLs (más basis risk).")
            calls = strikes_for(exch, args.underlying, code, "C")
            if not calls:
                raise RuntimeError("No hay CALLs para complemento.")
            k1 = None
            for sym, k in calls:
                if k >= args.strike:
                    k1 = (sym, k); break
            if k1 is None:
                k1 = calls[-1]
            target = k1[1] + args.dk
            k2 = min(calls, key=lambda x: abs(x[1]-target))
            if k2[1] <= k1[1]:
                bigger = [t for t in calls if t[1] > k1[1]]
                if not bigger:
                    raise RuntimeError("No hay strike superior.")
                k2 = bigger[0]
            Ck = price_from_symbol(k1[0])
            Ck2 = price_from_symbol(k2[0])
            dk = (k2[1] - k1[1])
            digital_up = (Ck - Ck2) / dk
            pB = (1.0 - digital_up) * args.payout
            pP = pm_mid * args.payout
            cost = cost_add(pP, pB, args.pm_bps, args.opt_bps, args.slip_bps)
            gap = pP - pB; edge = gap - cost
            res = {"price_PM": pP, "price_BIN": pB, "gap": gap, "cost": cost, "edge": edge}
            scen = scenarios_down(args.strike, k1[1], k2[1], args.payout)
            note = "Aprox DOWN via 1 - digital_up(K1,K2) con CALLs (más basis risk)."

    out = {
        "input": {
            "pm": {"token_id": tok, "mid_yes": pm_mid, "K_event": args.strike, "sense": args.sense},
            "binance": {"K1": k1[1], "K2": k2[1], "sym1": k1[0], "sym2": k2[0], "Δ": dk},
            "params": {"payout": args.payout, "fees_bps": {"pm": args.pm_bps, "opt": args.opt_bps, "slip": args.slip_bps}}
        },
        "results": res,
        "note": note,
        "scenarios": scen
    }
    if args.out == "csv":
        print("price_PM,price_BIN,edge")
        print(f"{res['price_PM']},{res['price_BIN']},{res['edge']}")
    else:
        txt = json.dumps(out, indent=2 if args.out=="pretty" else None, ensure_ascii=False)
        print(txt)

if __name__ == "__main__":
    main()
