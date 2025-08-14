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
import requests

PM_CLOB = "https://clob.polymarket.com"
PM_GAMMA = "https://gamma-api.polymarket.com"
BIN_EAPI = "https://eapi.binance.com"

DEFAULT_HEADERS = {"Accept": "application/json, text/plain, */*", "User-Agent": "curl/8.6.0"}

def http_get(url: str, params: dict = None, timeout: int = 20) -> Any:
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout); r.raise_for_status()
    try: return r.json()
    except Exception: return r.text

def http_post(url: str, payload: dict, timeout: int = 20) -> Any:
    r = requests.post(url, json=payload, headers=DEFAULT_HEADERS, timeout=timeout); r.raise_for_status()
    try: return r.json()
    except Exception: return r.text

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

# ---------- Menú ----------
def menu_select_asset() -> str:
    print("\nActivos disponibles:\n  [1] BTC\n  [2] ETH")
    while True:
        x = input("Elige activo [1/2]: ").strip()
        if x=="1": return "BTC"
        if x=="2": return "ETH"
        print("Opción inválida.")

def fetch_pm_candidates(asset: str) -> List[Dict[str,Any]]:
    # buscamos por slug amplio
    slugs = {
        "BTC": ["bitcoin","btc"],
        "ETH": ["ethereum","eth","ether"]
    }[asset]
    seen = {}
    out = []
    for s in slugs:
        data = clob_markets_by_slug(s)
        for m in data:
            q = (m.get("question") or "").lower()
            if asset=="BTC" and "bitcoin" not in q: continue
            if asset=="ETH" and ("ethereum" not in q and "ether" not in q and "eth " not in q): continue
            tokens = m.get("tokens") or []
            # outcomes tipo thresholds
            outs = [t.get("outcome","") for t in tokens if re.search(r'[<>]=?\s*\$?\d+(k|,?\d{3})?', t.get("outcome","").lower())]
            if not outs: continue
            cid = m.get("condition_id") or m.get("id")
            end_iso = m.get("end_date_iso") or m.get("end_date") or ""
            key = f"{cid}"
            if cid and key not in seen:
                seen[key] = True
                out.append({
                    "cid": cid,
                    "question": m.get("question"),
                    "end_date_iso": end_iso,
                    "market_slug": m.get("market_slug") or m.get("slug"),
                    "outcomes": outs
                })
    # ordena por fecha
    out.sort(key=lambda r: r.get("end_date_iso") or "")
    return out

def menu_select_pm_market(asset: str) -> Tuple[str, Dict[str,Any]]:
    cands = fetch_pm_candidates(asset)
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
            if not m: raise RuntimeError("CLOB no devolvió el mercado elegido.")
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

def run_menu():
    print("\n=== MODO MENÚ ===")
    asset = menu_select_asset()
    cid, market = menu_select_pm_market(asset)
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
    ap.add_argument("--menu", action="store_true", help="Lanza el menú interactivo (recomendado).")
    # modo antiguo por flags:
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
    args = ap.parse_args()

    if args.menu:
        return run_menu()

    # fallback: modo por flags (similar v2.8)
    if not (args.pm_token_id or args.pm_condition_id or args.pm_slug):
        raise SystemExit("Faltan parámetros. Usa --menu o pasa PM y Binance por flags.")
    if not (args.expiry and args.strike and args.sense):
        raise SystemExit("Debes pasar --expiry, --strike y --sense cuando no usas --menu.")

    # PM mid
    tok = args.pm_token_id
    if not tok:
        # resolución mínima por condition/slug+outcome
        if args.pm_condition_id:
            m = clob_market_by_condition(args.pm_condition_id)
            if not m: raise RuntimeError("No pude leer tokens para ese condition_id.")
            tokens = m.get("tokens") or []
            oq = normalize_outcome(args.pm_outcome or "")
            cand = [t for t in tokens if oq in normalize_outcome(t.get("outcome",""))]
            if not cand: raise RuntimeError("No encontré token para ese outcome.")
            tok = cand[0]["token_id"]
        else:
            # último recurso: buscar en slug de CLOB
            data = clob_markets_by_slug(args.pm_slug)
            if not data: raise RuntimeError("No encontré mercado para ese slug.")
            ymd = re.sub(r"\D", "", args.expiry)
            ymd = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}" if len(ymd)==8 else None
            targ = None
            for m in data:
                if ymd and (m.get("end_date_iso") or "").startswith(ymd): targ=m; break
            if not targ: targ = data[0]
            m2 = clob_market_by_condition(targ.get("condition_id"))
            if not m2: raise RuntimeError("No pude leer tokens del cid del slug.")
            oq = normalize_outcome(args.pm_outcome or "")
            cand = [t for t in (m2.get("tokens") or []) if oq in normalize_outcome(t.get("outcome",""))]
            if not cand: raise RuntimeError("No encontré token para ese outcome.")
            tok = cand[0]["token_id"]
    pm_mid = pm_mid_yes(tok)

    # Binance
    exch = load_exchange_info()
    code = expiry_to_code(args.expiry)
    if args.sense=="up":
        calls = strikes_for(exch, args.underlying, code, "C")
        if not calls: raise RuntimeError("No hay CALLs para esa expiración.")
        k1 = None
        for sym,k in calls:
            if k >= args.strike: k1=(sym,k); break
        if k1 is None: k1 = calls[-1]
        target = k1[1] + args.dk
        k2 = min(calls, key=lambda x: abs(x[1]-target))
        if k2[1] <= k1[1]:
            bigger = [t for t in calls if t[1] > k1[1]]
            if not bigger: raise RuntimeError("No hay strike superior para formar spread.")
            k2 = bigger[0]
        Ck  = price_from_symbol(k1[0])
        Ck2 = price_from_symbol(k2[0])
        dk  = (k2[1]-k1[1])
        res = compute_up(pm_mid, Ck, Ck2, dk, args.payout, args.pm_bps, args.opt_bps, args.slip_bps)
        scen = scenarios_up(args.strike, k1[1], k2[1], args.payout)
    else:
        puts = strikes_for(exch, args.underlying, code, "P")
        below_eq = [t for t in puts if t[1] <= args.strike]
        if not below_eq: raise RuntimeError("No hay PUTs ≤ K para esa expiración.")
        k1 = below_eq[-1]
        below = [t for t in puts if t[1] < k1[1]]
        if not below: raise RuntimeError("No hay strike inferior para put-spread.")
        target = k1[1] - args.dk
        k2 = min(below, key=lambda x: abs(x[1]-target))
        Pk  = price_from_symbol(k1[0])
        Pk2 = price_from_symbol(k2[0])
        dk  = (k1[1]-k2[1])
        res = compute_down(pm_mid, Pk, Pk2, dk, args.payout, args.pm_bps, args.opt_bps, args.slip_bps)
        scen = scenarios_down(args.strike, k1[1], k2[1], args.payout)

    out = {
        "input":{
            "pm":{"token_id": tok, "mid_yes": pm_mid, "K_event": args.strike, "sense": args.sense},
            "binance":{"K1": k1[1], "K2": k2[1], "sym1": k1[0], "sym2": k2[0], "Δ": dk},
            "params":{"payout": args.payout, "fees_bps":{"pm":args.pm_bps,"opt":args.opt_bps,"slip":args.slip_bps}}
        },
        "results": res,
        "scenarios": scen
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
