"""
Gold Analyzer v5 - วิเคราะห์ทอง XAUUSD (ไม่ต้องใช้ MT5)
v3: รวมเทคนิคเป็นระบบเดียว แบ่งเป็นชั้น (ภาวะตลาด ADX / ทิศทาง / โครงสร้าง / แรงส่ง / จุดเข้า / ยืนยัน)
v5 เพิ่ม: เกณฑ์ความน่าเชื่อถือจากผลย้อนหลัง (แยกช่วงทดสอบล่าสุด 30%), โหมดเรียลไทม์ไม่รอแท่งปิด,
  กรองข้อมูลเสีย/ทิกผิดปกติ, WebSocket ต่อใหม่อัตโนมัติ, กันแจ้งเตือนซ้ำ
v4 เพิ่ม:
  - ถือไม้สั้นสุด 1-3 นาที และโหมด "ต่อแท่ง" (เข้าหลังแท่งเทียนปิด ถือกี่แท่งก็ปิด)
  - ตัวจับเวลาปิดไม้ + นับถอยหลังแท่งปัจจุบัน
  - ผลย้อนหลังหักสเปรดแล้ว + เตือนเมื่อสเปรดกินระยะ SL มากเกินไป
รันด้วย: streamlit run app.py
"""
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Gold Analyzer", page_icon="🪙", layout="centered")

# ===================================================================================
# >>> CORE START  (ฟังก์ชันคำนวณล้วน ไม่พึ่ง Streamlit)
# ===================================================================================
TZ = "Asia/Bangkok"
BASE_COLS = ["open", "high", "low", "close"]
STEPS = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
         "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}

STRATS = {
    "pullback": "ย่อตามเทรนด์ (EMA21 + Fibonacci)",
    "breakout": "เบรกเอาต์ (Donchian 20)",
    "reversal": "กลับตัวที่ขอบ Bollinger (ไซด์เวย์)",
    "candle": "ต่อแท่ง: ตามโมเมนตัมแท่งเทียนที่เพิ่งปิด",
}
SHORT = {"pullback": "ย่อ", "breakout": "เบรก", "reversal": "กลับตัว", "candle": "ต่อแท่ง"}
KIND = {"pullback": "trend", "breakout": "trend", "reversal": "range", "candle": "trend"}
DEFAULT_STRATS = ("pullback", "breakout", "reversal")


def clean(df):
    """ทำความสะอาด OHLC + แปลงเวลาเป็นเวลาไทย (tz-aware)"""
    df = df.rename(columns=str.lower)[BASE_COLS].astype(float).dropna()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df > 0).all(axis=1) & (df.high >= df.low)]
    jump = df["close"].pct_change().abs().fillna(0)      # ตัดแท่งที่ราคากระโดดผิดปกติ (ข้อมูลเสีย)
    return df[jump < 0.06]


def resample_ohlc(df, rule):
    """รวมแท่งเล็กเป็นแท่งใหญ่ (เช่น 1h -> 4h)"""
    r = df[BASE_COLS].resample(rule)
    return pd.concat([r["open"].first(), r["high"].max(), r["low"].min(), r["close"].last()], axis=1).dropna()


def merge_tick(df, price, tick_time, step, max_gap=3):
    """รวมราคาทิกสดเข้ากับกราฟ คืน None ถ้าข้อมูลที่โหลดมาเก่าเกินไปเมื่อเทียบกับทิก"""
    last = df.index[-1]
    n = int((tick_time - last) // step)
    if n > max_gap:
        return None
    if n <= 0:
        df = df.copy()
        df.loc[last, "close"] = price
        df.loc[last, "high"] = max(df.loc[last, "high"], price)
        df.loc[last, "low"] = min(df.loc[last, "low"], price)
        return df
    o = float(df["close"].iloc[-1])
    new = pd.DataFrame({"open": [o], "high": [max(o, price)], "low": [min(o, price)], "close": [price]},
                       index=[last + step * n])
    return pd.concat([df, new])


def htf_flags(df, rule):
    """เทรนด์ของ Timeframe ใหญ่กว่า (EMA21/EMA50) ใช้เฉพาะแท่งใหญ่ที่ปิดแล้ว (ไม่แอบดูอนาคต)"""
    up = pd.Series(False, index=df.index)
    dn = pd.Series(False, index=df.index)
    if not rule:
        return up, dn, False
    h = resample_ohlc(df, rule)
    if len(h) < 60:
        return up, dn, False
    c = h["close"]
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    known = pd.DataFrame({"u": (c > e50) & (e21 > e50), "d": (c < e50) & (e21 < e50)})
    known.index = known.index + pd.Timedelta(rule)      # รู้ผลตอนแท่งใหญ่ปิด
    k = known.reindex(df.index, method="ffill")
    return k["u"].fillna(False).astype(bool), k["d"].fillna(False).astype(bool), True


def swing_pivots(h, l, k=3):
    """จุดสวิงที่ 'ยืนยันแล้ว' (ต้องมีแท่งปิดตามหลังอีก k แท่ง) ไม่รีเพนต์"""
    n = len(h)
    cH, pH, cL, pL = (np.full(n, np.nan) for _ in range(4))
    ch = ph = cl_ = pl = np.nan
    for i in range(n):
        j = i - k
        if j >= k:
            if h[j] == h[j - k:j + k + 1].max() and h[j] > h[j - 1]:
                ph, ch = ch, h[j]
            if l[j] == l[j - k:j + k + 1].min() and l[j] < l[j - 1]:
                pl, cl_ = cl_, l[j]
        cH[i], pH[i], cL[i], pL[i] = ch, ph, cl_, pl
    return cH, pH, cL, pL


def add_ind(df, htf_rule=None, min_score=60, enabled=DEFAULT_STRATS, live_last=False):
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["ema_t"] = c.ewm(span=200, adjust=False).mean()
    df["ema_s"] = c.ewm(span=50, adjust=False).mean()
    df["ema_f"] = c.ewm(span=21, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["rsi"] = rsi.where(loss != 0, np.where(gain > 0, 100.0, 50.0))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    # ความผันผวนต้องพอ (ATR ไม่ต่ำกว่า 60% ของค่ากลาง 100 แท่ง) ไม่งั้นสเปรดกินกำไร
    df["atr_ok"] = df["atr"] >= 0.6 * df["atr"].rolling(100, min_periods=20).median()

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    mh = macd - macd.ewm(span=9, adjust=False).mean()
    df["macd_h"] = mh
    m20, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    df["bb_m"], df["bb_u"], df["bb_l"] = m20, m20 + 2 * sd, m20 - 2 * sd

    um, dm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=df.index)
    mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    df["pdi"], df["mdi"] = pdi, mdi
    dh, dl = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)

    # แท่งเทียนกลับตัว/ยืนยัน: Engulfing / Hammer / Shooting star
    po, pcl = o.shift(1), c.shift(1)
    body, rng = (c - o).abs(), h - l
    lw = pd.concat([o, c], axis=1).min(axis=1) - l
    uw = h - pd.concat([o, c], axis=1).max(axis=1)
    bull_c = ((pcl < po) & (c > o) & (c >= po) & (o <= pcl)) | ((rng > 0) & (lw >= 2 * body) & (lw >= 0.55 * rng))
    bear_c = ((pcl > po) & (c < o) & (c <= po) & (o >= pcl)) | ((rng > 0) & (uw >= 2 * body) & (uw >= 0.55 * rng))
    df["bull_c"], df["bear_c"] = bull_c, bear_c
    bull_bar, bear_bar = c > o, c < o

    hu, hd, hok = htf_flags(df, htf_rule)
    df["htf_up"], df["htf_dn"], df["htf_ok"] = hu, hd, hok

    # ทิศทางหลัก
    up_s = (c > df.ema_t) & (df.ema_t > df.ema_t.shift(5))
    dn_s = (c < df.ema_t) & (df.ema_t < df.ema_t.shift(5))
    stk_u, stk_d = df.ema_f > df.ema_s, df.ema_f < df.ema_s
    up_t, dn_t = up_s & stk_u, dn_s & stk_d
    df["up_t"], df["dn_t"] = up_t, dn_t

    # โครงสร้างราคา (Swing) + Fibonacci
    cH, pH, cL, pL = (pd.Series(a, index=df.index) for a in
                      swing_pivots(h.to_numpy(float), l.to_numpy(float)))
    st_u, st_d = (cH > pH) & (cL > pL), (cH < pH) & (cL < pL)
    rsw = (cH - cL).where(cH > cL)
    fib_b = ((cH - l) / rsw).between(0.30, 0.80)       # ย่อลงมาลึก 30-80% ของขาขึ้นล่าสุด
    fib_s = ((h - cL) / rsw).between(0.30, 0.80)       # เด้งขึ้นมา 30-80% ของขาลงล่าสุด
    df["st_u"], df["st_d"], df["fib_b"], df["fib_s"] = st_u, st_d, fib_b, fib_s

    # แรงส่ง
    mac_u, mac_d = (mh > 0) & (mh > mh.shift(1)), (mh < 0) & (mh < mh.shift(1))
    df["mac_u"], df["mac_d"] = mac_u, mac_d
    adx = df["adx"]

    def I(s):
        return s.astype(int)

    # ---- คะแนนฝั่งตามเทรนด์ (รวม 100: ฐาน 90 + จุดเข้าเฉพาะกลยุทธ์ 10) ----
    base_b = (20 * I(up_s) + 5 * I(stk_u) + 20 * I(hu) + 10 * I(mac_u) + 5 * I(df.rsi.between(45, 68))
              + 10 * I((adx >= 20) & (pdi > mdi)) + 10 * I(bull_c) + 10 * I(st_u))
    base_s = (20 * I(dn_s) + 5 * I(stk_d) + 20 * I(hd) + 10 * I(mac_d) + 5 * I(df.rsi.between(32, 55))
              + 10 * I((adx >= 20) & (mdi > pdi)) + 10 * I(bear_c) + 10 * I(st_d))
    expand = rng > 1.1 * df.atr.shift(1)
    sc_pb_b = base_b + 5 * I(fib_b) + 5 * I(l <= df.ema_f)
    sc_pb_s = base_s + 5 * I(fib_s) + 5 * I(h >= df.ema_f)
    sc_bo_b = base_b + 5 * I(expand) + 5 * I(c > cH)
    sc_bo_s = base_s + 5 * I(expand) + 5 * I(c < cL)
    # โหมดต่อแท่ง: แท่งที่เพิ่งปิดต้องแรงและปิดใกล้ปลายแท่ง
    pos = (c - l) / rng.replace(0, np.nan)             # 0 = ปิดที่ต่ำสุดของแท่ง, 1 = ปิดที่สูงสุด
    sc_cd_b = base_b + 5 * I(pos >= 0.85) + 5 * I(expand)
    sc_cd_s = base_s + 5 * I(pos <= 0.15) + 5 * I(expand)
    # ---- คะแนนฝั่งไซด์เวย์ ----
    sc_rv_b = (25 * I(adx < 18) + 20 * I(df.rsi < 35) + 20 * I(l <= df.bb_l) + 15 * I(bull_c)
               + 10 * I(l <= cL + 0.5 * df.atr) + 10 * I(~hd))
    sc_rv_s = (25 * I(adx < 18) + 20 * I(df.rsi > 65) + 20 * I(h >= df.bb_u) + 15 * I(bear_c)
               + 10 * I(h >= cH - 0.5 * df.atr) + 10 * I(~hu))

    # ---- จุดเข้า: แยกตามภาวะตลาด (ADX>=20 เทรนด์ / ADX<18 ไซด์เวย์ / 18-20 ไม่เทรด) ----
    trg = {
        "pullback": (up_t & (adx >= 20) & (l <= df.ema_f) & (c > df.ema_f) & (c > df.ema_s) & bull_bar
                     & df.rsi.between(40, 68),
                     dn_t & (adx >= 20) & (h >= df.ema_f) & (c < df.ema_f) & (c < df.ema_s) & bear_bar
                     & df.rsi.between(32, 60)),
        "breakout": (up_t & (c > dh) & (adx >= 20) & bull_bar & ((c - o) > 0.5 * df.atr)
                     & ((c - df.ema_f) < 2.5 * df.atr) & (df.rsi < 75),
                     dn_t & (c < dl) & (adx >= 20) & bear_bar & ((o - c) > 0.5 * df.atr)
                     & ((df.ema_f - c) < 2.5 * df.atr) & (df.rsi > 25)),
        "reversal": ((adx < 18) & (l <= df.bb_l) & (c > df.bb_l) & (df.rsi < 40) & bull_bar,
                     (adx < 18) & (h >= df.bb_u) & (c < df.bb_u) & (df.rsi > 60) & bear_bar),
        "candle": (up_t & (adx >= 18) & bull_bar & ((c - o) >= 0.6 * df.atr) & (pos >= 0.7) & (c > df.ema_f)
                   & mac_u & df.rsi.between(50, 74) & ((c - df.ema_f) < 2.0 * df.atr),
                   dn_t & (adx >= 18) & bear_bar & ((o - c) >= 0.6 * df.atr) & (pos <= 0.3) & (c < df.ema_f)
                   & mac_d & df.rsi.between(26, 50) & ((df.ema_f - c) < 2.0 * df.atr)),
    }
    scr = {"pullback": (sc_pb_b, sc_pb_s), "breakout": (sc_bo_b, sc_bo_s), "reversal": (sc_rv_b, sc_rv_s),
           "candle": (sc_cd_b, sc_cd_s)}

    gate_b = df["atr_ok"] & ~hd         # ห้ามสวน TF ใหญ่ + ตลาดต้องไม่นิ่งเกิน
    gate_s = df["atr_ok"] & ~hu
    n = len(df)
    anyb, anys = np.zeros(n, bool), np.zeros(n, bool)
    bsc, ssc = np.zeros(n), np.zeros(n)
    for k in STRATS:
        tb, ts = trg[k]
        sb_, ss_ = scr[k]
        df[f"sc_{k}_b"], df[f"sc_{k}_s"] = sb_, ss_
        buy = np.array(tb & gate_b & (sb_ >= min_score), dtype=bool)
        sell = np.array(ts & gate_s & (ss_ >= min_score), dtype=bool)
        if not live_last:
            buy[-1] = sell[-1] = False                   # แท่งสุดท้ายยังไม่ปิด ไม่นับ (โหมดเรียลไทม์จะนับ)
        if k not in enabled:                             # กลยุทธ์ที่ไม่ได้ใช้ในโปรไฟล์นี้
            buy[:] = False
            sell[:] = False
        df[f"{k}_buy"], df[f"{k}_sell"] = buy, sell
        anyb |= buy
        anys |= sell
        bsc = np.maximum(bsc, np.where(buy, sb_.to_numpy(float), 0))
        ssc = np.maximum(ssc, np.where(sell, ss_.to_numpy(float), 0))
    df["buy_sig"], df["sell_sig"] = anyb, anys
    df["buy_sc"], df["sell_sc"] = bsc, ssc
    return df


def swing_levels(df, k=5, look=300):
    d = df.tail(look)
    hi, lo = d["high"].values, d["low"].values
    res, sup = [], []
    for i in range(k, len(d) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            res.append(hi[i])
        if lo[i] == lo[i - k:i + k + 1].min():
            sup.append(lo[i])
    return sup, res


def day_levels(df):
    """สูง/ต่ำของวันก่อนหน้า (แนวสำคัญที่คนใช้กันเยอะ)"""
    try:
        dd = resample_ohlc(df, "1D")
        if len(dd) >= 2:
            return float(dd["high"].iloc[-2]), float(dd["low"].iloc[-2])
    except Exception:
        pass
    return None, None


def mults(prof, k):
    """ตัวคูณ ATR (SL, TP1): กลับตัวใช้เป้าสั้นกว่า เบรกเอาต์ใช้เป้ายาวกว่า"""
    sl, tp = prof["sl"], prof["tp"]
    return {"pullback": (sl, tp), "breakout": (sl, tp * 1.2), "reversal": (sl * 0.8, tp * 0.6),
            "candle": (sl, tp)}[k]


def analyse(df, prof, min_score=60, spread=0.0, intrabar=False):
    enabled = tuple(prof.get("strats", DEFAULT_STRATS))
    cand_only = enabled == ("candle",)
    d = add_ind(df.tail(2500).copy(), prof.get("htf"), min_score, enabled, live_last=intrabar)
    si = -1 if intrabar else -2                    # แท่งสัญญาณ: เรียลไทม์ = แท่งที่กำลังวิ่ง / ปกติ = แท่งที่ปิดแล้วล่าสุด
    b1 = d.iloc[si]
    price = float(d.iloc[-1]["close"])
    atr = float(b1["atr"])
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("ค่า ATR ใช้ไม่ได้ (ข้อมูลนิ่งหรือไม่พอ)")
    up, dn = bool(b1.up_t), bool(b1.dn_t)
    bias = "up" if up else "down" if dn else "side"
    adx = float(b1.adx)
    regime = "trend" if adx >= 20 else "range" if adx < 18 else "mixed"
    atr_ok = bool(b1.atr_ok)
    htf_ok = bool(b1.htf_ok)

    fired = []
    for k in enabled:
        if bool(d[f"{k}_buy"].iloc[si]):
            fired.append((k, "buy", float(b1[f"sc_{k}_b"])))
        if bool(d[f"{k}_sell"].iloc[si]):
            fired.append((k, "sell", float(b1[f"sc_{k}_s"])))
    fired.sort(key=lambda x: -x[2])

    sup, res = swing_levels(d)
    pdh, pdl = day_levels(df)
    if pdh is not None:
        res.append(pdh)
        sup.append(pdl)
    near_res = min([x for x in res if x > price], default=None)
    near_sup = max([x for x in sup if x < price], default=None)

    side, strat, score, status, msg = None, None, 0.0, "no", ""
    lo = hi = price
    if fired:
        strat, side, score = fired[0]
        ref = float(b1.close)
        za, zb = (0.15, 0.15) if cand_only else (0.35, 0.15)     # โหมดต่อแท่ง: เข้าที่ราคาตลาดหลังแท่งปิด โซนแคบกว่า
        late = 0.25 if cand_only else 0.4
        if side == "buy":
            lo, hi = ref - za * atr, ref + zb * atr
            if (not intrabar) and price < float(b1.low) - 0.1 * atr:
                status, msg = "dead", "ราคาหลุดต่ำกว่าแท่งสัญญาณแล้ว สัญญาณ BUY ถูกยกเลิก อย่าเข้า"
            elif price > hi + late * atr:
                status, msg = "late", "ราคาวิ่งหนีจากจุดเข้าไปแล้ว อย่าไล่ รอสัญญาณใหม่"
            else:
                status = "now"
        else:
            lo, hi = ref - zb * atr, ref + za * atr
            if (not intrabar) and price > float(b1.high) + 0.1 * atr:
                status, msg = "dead", "ราคาทะลุเหนือแท่งสัญญาณแล้ว สัญญาณ SELL ถูกยกเลิก อย่าเข้า"
            elif price < lo - late * atr:
                status, msg = "late", "ราคาวิ่งหนีจากจุดเข้าไปแล้ว อย่าไล่ รอสัญญาณใหม่"
            else:
                status = "now"
        if status != "now":
            side = None
    elif cand_only:
        msg = ("ตลาดนิ่งเกินไป (ATR ต่ำ) สเปรดจะกินกำไร" if not atr_ok
               else "ยังไม่มีแท่งเทียนที่ปิดด้วยโมเมนตัมแรงตามเทรนด์ รอแท่งถัดไป")
    elif not atr_ok:
        msg = "ตลาดนิ่งเกินไป (ATR ต่ำ) สเปรดจะกินกำไร"
    elif regime == "mixed":
        msg = f"ADX {adx:.0f} อยู่ช่วงกำกวม (18-20) ยังบอกไม่ได้ว่าเป็นเทรนด์หรือไซด์เวย์ ไม่เทรด"
    elif regime == "trend":
        if not (up or dn):
            msg = "ADX บอกว่ามีเทรนด์ แต่ EMA ยังไม่เรียงทิศชัดเจน รอก่อน"
        elif (up and htf_ok and b1.htf_dn) or (dn and htf_ok and b1.htf_up):
            msg = f"ทิศทางสวนเทรนด์ TF ใหญ่ ({prof.get('htf')}) ไม่เข้าไม้"
        else:
            ema_f, ema_s = float(b1.ema_f), float(b1.ema_s)
            zl, zh = (ema_f - 0.3 * atr, ema_f + 0.1 * atr) if up else (ema_f - 0.1 * atr, ema_f + 0.3 * atr)
            if up and price < ema_s:
                msg = "ราคาหลุด EMA50 เทรนด์ขึ้นเริ่มเสีย ห้ามรับ BUY"
            elif dn and price > ema_s:
                msg = "ราคาทะลุ EMA50 เทรนด์ลงเริ่มเสีย ห้ามเข้า SELL"
            elif abs(price - ema_f) > 2.5 * atr and ((up and price > zh) or (dn and price < zl)):
                msg = "ราคาอยู่ไกลจากจุดย่อ/เด้งมาก อย่าไล่ รอให้ย้อนกลับมาที่ EMA21"
            elif (up and price < zl) or (dn and price > zh):
                msg = "ราคาทะลุผ่านโซนย่อ/เด้งไปแล้ว รอดูว่าจะยืนกลับมาได้ไหม"
            else:
                side, strat, status = ("buy" if up else "sell"), "pullback", "wait"
                score = float(b1["sc_pullback_b"] if up else b1["sc_pullback_s"])
                lo, hi = zl, zh
    elif regime == "range" and np.isfinite(b1.bb_l):
        near_low = abs(price - b1.bb_l) <= abs(b1.bb_u - price)
        band = float(b1.bb_l if near_low else b1.bb_u)
        if abs(price - band) <= 1.0 * atr and not (near_low and htf_ok and b1.htf_dn) \
                and not ((not near_low) and htf_ok and b1.htf_up):
            side, strat, status = ("buy" if near_low else "sell"), "reversal", "wait"
            score = float(b1["sc_reversal_b"] if near_low else b1["sc_reversal_s"])
            lo, hi = ((band - 0.1 * atr, band + 0.3 * atr) if near_low else (band - 0.3 * atr, band + 0.1 * atr))
        else:
            msg = "ตลาดไซด์เวย์ แต่ราคายังอยู่กลางกรอบ (หรือสวน TF ใหญ่) ยังไม่ใกล้ขอบ Bollinger"
    if not msg and status == "no":
        msg = "ยังไม่เข้าเงื่อนไขของกลยุทธ์ใดเลย"

    plan, warns = None, []
    if side:
        sl_m, tp_m = mults(prof, strat)
        mid = (lo + hi) / 2
        s = 1 if side == "buy" else -1
        sl = mid - s * sl_m * atr
        tp1 = mid + s * tp_m * atr
        tp2 = mid + s * tp_m * 1.6 * atr
        sl_dist, tp_dist = abs(mid - sl), abs(tp1 - mid)
        sp_pct = 100 * spread / sl_dist if spread > 0 else 0.0
        rr_net = max(0.0, tp_dist - spread) / (sl_dist + spread)
        # เวลาปิดไม้ตามเวลาถือ: แท่งสัญญาณปิดแล้วเข้า ถือครบ max_bars แท่งแล้วปิดที่ราคาตลาด
        if status != "now":
            exit_at = None
        elif intrabar:                              # เข้าตอนนี้ กลางแท่ง ถือครบ max_bars แท่งนับจากแท่งปัจจุบัน
            exit_at = d.index[-1] + prof["step"] * prof["max_bars"]
        else:
            exit_at = d.index[-2] + prof["step"] * (prof["max_bars"] + 1)
        plan = dict(side=side, lo=lo, hi=hi, mid=mid, sl=sl, tp1=tp1, tp2=tp2, sl_dist=sl_dist,
                    rr=tp_m / sl_m, rr_net=rr_net, sp_pct=sp_pct, exit_at=exit_at, strat=strat, score=score)
        if sp_pct >= 35:
            warns.append(f"สเปรดกิน {sp_pct:.0f}% ของระยะ SL ต้นทุนสูงเกินไป โอกาสกำไรสุทธิต่ำมาก ไม่แนะนำเข้า "
                         "(ถ้าอยากเทรดสั้นแบบนี้ ต้องใช้โบรกเกอร์ที่สเปรดแคบมาก)")
        elif sp_pct >= 20:
            warns.append(f"สเปรดกิน {sp_pct:.0f}% ของระยะ SL RR สุทธิเหลือ 1 : {rr_net:.2f} ต้องชนะบ่อยกว่าปกติถึงจะคุ้ม")
        if side == "buy" and near_res and near_res < tp1:
            warns.append(f"มีแนวต้านที่ {near_res:.2f} ก่อนถึง TP1 ราคาอาจชนแนวนี้ก่อน (พิจารณาออกบางส่วน)")
        if side == "sell" and near_sup and near_sup > tp1:
            warns.append(f"มีแนวรับที่ {near_sup:.2f} ก่อนถึง TP1 ราคาอาจชนแนวนี้ก่อน (พิจารณาออกบางส่วน)")
        if side == "buy" and b1.rsi > 70:
            warns.append("RSI สูง ระวังไล่ราคา")
        if side == "sell" and b1.rsi < 30:
            warns.append("RSI ต่ำ ระวังไล่ราคา")
        if prof.get("scalp") and 3 <= d.index[-1].hour < 14:
            warns.append("ช่วงเวลาเอเชีย (ไทย 03:00-14:00) ทองผันผวนต่ำ ไม้สั้นสเปรดกินกำไรง่าย")

    cs = side or ("buy" if up else "sell" if dn else None)

    def pick(a, b):
        return bool(a) if cs == "buy" else bool(b) if cs == "sell" else False

    checks = {
        "เทรนด์หลัก EMA200 ชัดเจน (ราคาเทียบ EMA200 และเส้นชี้ทิศ)": up or dn,
        "EMA21 / EMA50 เรียงตามทิศทาง": pick(b1.ema_f > b1.ema_s, b1.ema_f < b1.ema_s),
        (f"TF ใหญ่ ({prof.get('htf')}) เห็นด้วย" if htf_ok else "TF ใหญ่: ข้อมูลไม่พอ"): pick(b1.htf_up, b1.htf_dn),
        "โครงสร้างราคา (HH/HL สำหรับขึ้น, LH/LL สำหรับลง)": pick(b1.st_u, b1.st_d),
        "MACD ไปทางเดียวกันและแรงขึ้นต่อ": pick(b1.mac_u, b1.mac_d),
        f"RSI อยู่ในช่วงเหมาะสม (ตอนนี้ {b1.rsi:.0f})": pick(45 <= b1.rsi <= 68, 32 <= b1.rsi <= 55),
        f"ADX {adx:.0f} (ตั้งแต่ 20 = มีเทรนด์)": regime == "trend",
        "แท่งเทียนยืนยันทิศทาง (Engulfing, Hammer, Shooting star)": pick(b1.bull_c, b1.bear_c),
        "ราคาอยู่ในโซน Fibonacci 38-62% ของขาล่าสุด": pick(b1.fib_b, b1.fib_s),
        "ความผันผวนพอ (ATR ไม่นิ่งเกินไป)": atr_ok,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(d=d, bias=bias, status=status, msg=msg, plan=plan, price=price, atr=atr, checks=checks,
                sup=near_sup, res=near_res, warns=warns, bar=d.index[si], fired=fired, intrabar=intrabar,
                regime=regime, htf_ok=htf_ok, enabled=enabled, cand_only=cand_only)


def _sim(d, events, max_bars, cost=0.0):
    """จำลองรายไม้: เข้าที่ราคาปิดแท่งสัญญาณ ออกที่ SL / TP1 / หมดเวลาถือ (ปิดที่ราคาตลาด)
    ชนทั้ง SL และ TP ในแท่งเดียว = แพ้ ไม่เปิดไม้ซ้อน หักต้นทุนสเปรด (cost ดอลลาร์/ไม้) แล้ว"""
    hi, lo, cl, atr = (d[c].to_numpy(float) for c in ("high", "low", "close", "atr"))
    n, wins, losses, r_sum, free_from, timeouts = len(d), 0, 0, 0.0, 0, 0
    for i, s, sl_m, tp_m in events:
        if i < free_from or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        sl, tp = cl[i] - s * sl_m * atr[i], cl[i] + s * tp_m * atr[i]
        end = min(n - 1, i + max_bars)
        out = None
        for j in range(i + 1, end + 1):
            hit_sl = lo[j] <= sl if s == 1 else hi[j] >= sl
            hit_tp = hi[j] >= tp if s == 1 else lo[j] <= tp
            if hit_sl or hit_tp:
                out = (j, -1.0 if hit_sl else tp_m / sl_m)
                break
        if out is None:
            if i + max_bars > n - 1:
                continue                                   # ไม้ยังไม่ครบเวลาถือ ไม่นับ
            out = (end, s * (cl[end] - cl[i]) / (sl_m * atr[i]))
            timeouts += 1
        j, r = out
        r -= cost / (sl_m * atr[i])                        # หักสเปรดเป็นหน่วย R
        if r > 0:
            wins += 1
        else:
            losses += 1
        r_sum += r
        free_from = j + 1
    total = wins + losses
    return dict(n=total, wins=wins, losses=losses, winrate=(100 * wins / total) if total else 0.0,
                exp_r=(r_sum / total) if total else 0.0, timeouts=timeouts)


def grade(r):
    """ประเมินความน่าเชื่อถือจากผลย้อนหลังหักสเปรด: pass / few (ไม้น้อยเกินสรุปไม่ได้) / fail"""
    if r["n"] < 20 or r["oos"]["n"] < 8:
        return "few"
    return "pass" if (r["exp_r"] > 0.05 and r["oos"]["exp_r"] > 0) else "fail"


def _split(d, ev, max_bars, cost, cut):
    full = _sim(d, ev, max_bars, cost)
    full["is"] = _sim(d, [e for e in ev if e[0] < cut], max_bars, cost)       # 70% แรก
    full["oos"] = _sim(d, [e for e in ev if e[0] >= cut], max_bars, cost)     # 30% ล่าสุด (ไม่ได้ใช้ปรับค่า)
    full["grade"] = grade(full)
    return full


def backtest(d, prof, cost=0.0):
    """ผลย้อนหลังแยกรายกลยุทธ์ที่ใช้ + รวม (คีย์ 'all') หักสเปรดแล้ว แยกช่วงทดสอบล่าสุด 30% ('oos') ยังไม่รวมสลิป"""
    n, max_bars = len(d), prof["max_bars"]
    cut = int(n * 0.7)
    ens = tuple(prof.get("strats", DEFAULT_STRATS))
    sig = {k: (d[f"{k}_buy"].to_numpy(bool), d[f"{k}_sell"].to_numpy(bool)) for k in STRATS}
    out = {}
    for k in ens:
        b, s_ = sig[k]
        sl_m, tp_m = mults(prof, k)
        out[k] = _split(d, [(i, 1 if b[i] else -1, sl_m, tp_m) for i in np.flatnonzero(b | s_)], max_bars, cost, cut)
    anyf = np.zeros(n, bool)
    for k in ens:
        anyf |= sig[k][0] | sig[k][1]
    scb = {k: d[f"sc_{k}_b"].to_numpy(float) for k in ens}
    scs = {k: d[f"sc_{k}_s"].to_numpy(float) for k in ens}
    ev = []
    for i in np.flatnonzero(anyf):
        best = None
        for k in ens:
            if sig[k][0][i] and (best is None or scb[k][i] > best[0]):
                best = (scb[k][i], 1, k)
            if sig[k][1][i] and (best is None or scs[k][i] > best[0]):
                best = (scs[k][i], -1, k)
        sl_m, tp_m = mults(prof, best[2])
        ev.append((i, best[1], sl_m, tp_m))
    out["all"] = _split(d, ev, max_bars, cost, cut)
    return out
# ===================================================================================
# <<< CORE END
# ===================================================================================

# ระยะเวลาถือไม้ -> กราฟที่ใช้วิเคราะห์ / TF ใหญ่ที่ใช้ยืนยันเทรนด์ / ตัวคูณ ATR สำหรับ SL, TP
# max_bars = เวลาถือสูงสุดเป็นจำนวนแท่ง (ครบแล้วปิดไม้ที่ราคาตลาด) / strats = กลยุทธ์ที่ใช้ในโปรไฟล์นี้
PROFILES = {
    "ต่อแท่ง M1 · ถือ 1-3 นาที": dict(
        tf="M1", interval="1m", period="7d", sl=1.0, tp=1.2, htf="15min", max_bars=3, scalp=True,
        hold="ถือ 1-3 นาที (เข้าหลังแท่ง M1 ปิด ถือ 3 แท่ง)", strats=("candle",)),
    "ต่อแท่ง M5 · ถือ 5-15 นาที": dict(
        tf="M5", interval="5m", period="59d", sl=1.0, tp=1.3, htf="1h", max_bars=3, scalp=True,
        hold="ถือ 5-15 นาที (เข้าหลังแท่ง M5 ปิด ถือ 3 แท่ง)", strats=("candle",)),
    "สกัลป์สั้น · ถือ 3-10 นาที": dict(
        tf="M1", interval="1m", period="7d", sl=1.0, tp=1.5, htf="15min", max_bars=10, scalp=True,
        hold="ถือราว 3-10 นาที"),
    "สกัลป์ · ถือ 5-30 นาที": dict(
        tf="M1", interval="1m", period="7d", sl=1.2, tp=1.8, htf="15min", max_bars=30, scalp=True,
        hold="ถือราว 5-30 นาที"),
    "ไม้สั้น · ถือ 30 นาที-3 ชม.": dict(
        tf="M5", interval="5m", period="59d", sl=1.5, tp=2.5, htf="1h", max_bars=36, scalp=True,
        hold="ถือราว 30 นาที-3 ชั่วโมง"),
    "ไม้กลาง · ถือ 3-12 ชม. (ในวัน)": dict(
        tf="M15", interval="15m", period="59d", sl=1.5, tp=3.0, htf="4h", max_bars=48, scalp=False,
        hold="ถือราว 3-12 ชั่วโมง (จบในวัน)"),
    "ไม้ยาว · ถือ 1-3 วัน": dict(
        tf="H1", interval="1h", period="180d", sl=1.5, tp=3.0, htf="4h", max_bars=72, scalp=False,
        hold="ถือราว 1-3 วัน"),
    "สวิง · ถือหลายวัน-สัปดาห์": dict(
        tf="H4", interval="1h", period="365d", resample="4h", sl=1.8, tp=4.0, htf="1D", max_bars=60, scalp=False,
        hold="ถือหลายวันถึงประมาณ 1-2 สัปดาห์"),
}
DEFAULT_HOLD = "ไม้สั้น · ถือ 30 นาที-3 ชม."
SYMBOLS = {"XAUUSD=X (ทองสปอต)": "XAUUSD=X", "GC=F (ทองฟิวเจอร์ส COMEX)": "GC=F"}
SPOT = "XAUUSD=X"


def _secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def fmt_td(td):
    s = max(0, int(td.total_seconds()))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("ตั้งค่า")
sym_label = sb.selectbox("แหล่งราคา", list(SYMBOLS),
                         help="MT5 ส่วนใหญ่เทรดทองสปอต เลือก XAUUSD=X จะใกล้เคียงที่สุด")
hold_names = list(PROFILES)
hold_key = sb.selectbox("ระยะเวลาถือไม้", hold_names, index=hold_names.index(DEFAULT_HOLD),
                        help="ระบบจะปรับ Timeframe, เป้า TP/SL, เวลาปิดไม้ และ TF ใหญ่ที่ใช้ยืนยันเทรนด์ให้เอง "
                             "โหมด 'ต่อแท่ง' = รอแท่งเทียนปิดแรงตามเทรนด์ แล้วเข้าที่แท่งถัดไป ถือครบ 3 แท่งปิดที่ราคาตลาด "
                             "ถือสั้นมากสเปรดกินกำไรง่าย ดูผลย้อนหลังหักสเปรดในกล่อง 'เปรียบเทียบกลยุทธ์' ก่อนเสมอ")
P = dict(PROFILES[hold_key], step=STEPS[PROFILES[hold_key].get("resample", PROFILES[hold_key]["interval"])])
tf = P["tf"]
sb.caption(f"วิเคราะห์จากกราฟ {tf} | ยืนยันเทรนด์ด้วย {P['htf']} | {P['hold']}")
min_score = sb.slider("ความเข้มงวด (คะแนนขั้นต่ำ 0-100)", 40, 90, 60, 5,
                      help="สัญญาณต้องผ่านคะแนนรวมของทุกชั้นวิเคราะห์เท่านี้ขึ้นไปถึงจะแสดง ยิ่งสูงยิ่งสัญญาณน้อยแต่คัดมาแล้ว")
spread_usd = sb.number_input("สเปรดของโบรกเกอร์ ($ ต่อไม้)", value=0.30, min_value=0.0, max_value=5.0, step=0.05,
                             help="ดูจาก MT5 (ราคา Ask - Bid) ใช้หักต้นทุนในผลย้อนหลัง และเตือนเมื่อสเปรดกินระยะ SL มากเกินไป")
intrabar = sb.checkbox("⚡ เรียลไทม์: ไม่ต้องรอแท่งปิด (สัญญาณชั่วคราว)", value=False,
                       help="เปิด = ประเมินสัญญาณจากแท่งที่กำลังวิ่งอยู่ทันที ได้จังหวะเร็วกว่า แต่สัญญาณอาจหายไปก่อนแท่งปิด "
                            "(ราคาย้อนกลับ) ความแม่นยำต่ำกว่าแบบรอแท่งปิด และผลย้อนหลังด้านล่างคำนวณจากแท่งที่ปิดแล้วเท่านั้น "
                            "ดูสถิติ 'ยืนยันเมื่อแท่งปิด' ในกล่องเรียลไทม์ประกอบ")
gate = sb.checkbox("ซ่อนคำสั่งเข้าไม้ถ้ากติกายังไม่ผ่านเกณฑ์ย้อนหลัง", value=True,
                   help="เปิดอยู่ = จะแสดงว่า 'เข้าได้ตอนนี้' และส่งแจ้งเตือนต่อเมื่อกลยุทธ์นั้นมีผลย้อนหลังหักสเปรดเป็นบวก "
                        "ทั้งช่วงรวมและช่วงทดสอบล่าสุด 30% (ไม้ ≥20 และ ≥8 ตามลำดับ) แนะนำให้เปิดไว้")
sb.subheader("แจ้งเตือน Telegram (ไม่บังคับ)")
tg_token = sb.text_input("Bot token", value=_secret("TG_TOKEN"), type="password")
tg_chat = sb.text_input("Chat ID", value=str(_secret("TG_CHAT")))
speed = sb.selectbox("ความเร็วอัปเดตกราฟ (วินาที)", [0.5, 1, 2, 5, 10, 30, 60], index=1,
                     help="Yahoo ส่งทิกราว 1 ครั้ง/วินาที เลือก 1-2 วินาทีก็พอ ถ้าเครื่องหรือเน็ตช้าให้เลือกมากกว่านี้")


# ---------------- Telegram ----------------
def send_tg(msg, token, chat):
    """คืน (สำเร็จไหม, ข้อความผลลัพธ์) ไม่โยน exception และไม่เอา token ไปใส่ในข้อความ error"""
    if not (token and chat):
        return False, "ยังไม่ได้ใส่ Bot token / Chat ID"
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": msg}, timeout=10)
        if r.ok:
            return True, "ส่งสำเร็จ"
        try:
            why = r.json().get("description", "")
        except Exception:
            why = ""
        return False, f"Telegram ตอบ {r.status_code} {why}".strip()
    except Exception as e:
        return False, f"ส่งไม่สำเร็จ ({type(e).__name__})"


if sb.button("ส่งข้อความทดสอบ Telegram"):
    ok, info = send_tg("Gold Analyzer: ทดสอบการแจ้งเตือน ✅", tg_token, tg_chat)
    (sb.success if ok else sb.error)(info)


# ---------------- Data ----------------
@st.cache_data(ttl=5, show_spinner=False)
def load_yahoo(sym, interval, period):
    """คืน (df หรือ None, ข้อความ) รีเฟรชได้ทุก ~5 วินาที"""
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and len(df) >= 260:
            return clean(df), ""
        return None, f"Yahoo {sym}: ได้ {0 if df is None else len(df)} แท่ง"
    except Exception as e:
        return None, f"Yahoo {sym}: {type(e).__name__}"


def load(symbol, interval, period, resample=None):
    """ลองแหล่งที่เลือกก่อน ถ้าไม่ได้ลองอีกตัว คืน (df, ข้อความ, สัญลักษณ์ที่ใช้จริง)"""
    other = "GC=F" if symbol == SPOT else SPOT
    notes = []
    for code in (symbol, other):
        df, note = load_yahoo(code, interval, period)
        if df is not None and resample:
            df = resample_ohlc(df, resample)
            if len(df) < 260:
                notes.append(f"{code}: รวมเป็น {resample} ได้ {len(df)} แท่ง ไม่พอ")
                continue
        if df is not None:
            return df, f"ใช้ข้อมูล Yahoo ({code})" + ("" if code == symbol else " แทนแหล่งที่เลือก"), code
        notes.append(note)
    return None, " | ".join(n for n in notes if n), None


LIVE = {}   # symbol -> {"price": ราคาล่าสุด, "ts": เวลาที่รับ}
WS = {"err": None}


@st.cache_resource
def get_alerted():
    return {"keys": []}          # กันแจ้งเตือนซ้ำข้ามเซสชัน/แท็บ


@st.cache_resource
def get_iblog():
    return {}                    # บันทึกสัญญาณเรียลไทม์ -> ยืนยันหรือไม่เมื่อแท่งปิด


@st.cache_resource
def start_stream(symbol):
    """เปิด WebSocket ของ Yahoo ไว้เบื้องหลัง รับราคาทิกสดเก็บไว้ใน LIVE (เปิดครั้งเดียวต่อสัญลักษณ์)"""
    if not hasattr(yf, "WebSocket"):
        WS["err"] = "ไลบรารีไม่รองรับ"
        return None

    def handler(msg):
        try:
            if msg.get("id") not in (None, symbol):
                return
            price = msg.get("price")
            if price is None:
                return
            price = float(price)
            prev = LIVE.get(symbol)
            if price <= 0 or (prev and time.time() - prev["ts"] < 60 and abs(price / prev["price"] - 1) > 0.03):
                return                                   # ทิกผิดปกติ (กระโดด >3%) ทิ้ง
            LIVE[symbol] = {"price": price, "ts": time.time()}
        except Exception:
            pass

    def run():
        wait = 5
        while True:
            t0 = time.time()
            try:
                with yf.WebSocket() as ws:
                    WS["err"] = None
                    ws.subscribe([symbol])
                    ws.listen(handler)
            except Exception as e:
                WS["err"] = type(e).__name__
            wait = 5 if time.time() - t0 > 60 else min(60, wait * 2)    # ต่อใหม่ ถ้าหลุดบ่อยจะรอนานขึ้น
            time.sleep(wait)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ---------------- UI helpers ----------------
_NEW_ST = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


def stretch(fn, *args, **kw):
    """เรียก st.dataframe / st.plotly_chart ให้เต็มความกว้าง รองรับ Streamlit ทุกเวอร์ชัน
    (ถ้าเวอร์ชันไหนไม่รู้จักพารามิเตอร์ ก็ลองแบบถัดไปโดยอัตโนมัติ)"""
    widths = ([dict(width="stretch")] if _NEW_ST else []) + [dict(use_container_width=True)]
    for extra in widths:
        for drop_key in (False, True):
            kw2 = {a: b for a, b in kw.items() if not (drop_key and a == "key")}
            try:
                return fn(*args, **extra, **kw2)
            except TypeError:
                continue
    return fn(*args)


# ---------------- UI ----------------
st.title("🪙 Gold Analyzer")
st.caption("ไม่มีระบบเทรดใดแม่นยำ 100% แอปนี้จึงแสดงสถิติย้อนหลังจริง (หักสเปรด แยกช่วงทดสอบ) ให้คุณตัดสินใจเอง "
           "และซ่อนคำสั่งเข้าไม้เมื่อกติกายังไม่ผ่านเกณฑ์ ตรวจกราฟจริงและตั้ง SL ทุกไม้")

# CSS ทำครั้งเดียวนอก fragment: ไม่ให้ element จางลงตอนรีเฟรช
st.markdown("""<style>
[data-stale="true"], .stale-element,
[data-testid="stElementContainer"][data-stale="true"],
[data-testid="stPlotlyChart"] {opacity:1 !important; transition:none !important;}
</style>""", unsafe_allow_html=True)


@st.fragment(run_every=speed)
def live_view():
    chart_box = st.container()      # ตำแหน่งกราฟคงที่ ไม่ถูก mount ใหม่
    sym = SYMBOLS[sym_label]
    start_stream(sym)
    key = (sym, hold_key)
    df, src_note, src = load(sym, P["interval"], P["period"], P.get("resample"))
    data_ok = df is not None
    if data_ok:
        st.session_state["last_good"] = dict(key=key, df=df, note=src_note, src=src, t=time.time())
    else:
        lg = st.session_state.get("last_good")
        if lg and lg["key"] == key and time.time() - lg["t"] < 600:
            df, src, src_note = lg["df"], lg["src"], lg["note"] + " (ข้อมูลเก่าจากรอบก่อน)"
            st.warning("ดึงข้อมูลใหม่ไม่ได้ชั่วคราว กำลังใช้ข้อมูลรอบก่อนหน้า ยังไม่ควรเปิดไม้จากสัญญาณนี้")
        else:
            st.error("ดึงข้อมูลราคาไม่ได้ในตอนนี้")
            st.code(src_note or "ไม่มีรายละเอียด")
            st.info("Yahoo อาจปิดกั้นชั่วคราว (โดยเฉพาะบนเซิร์ฟเวอร์คลาวด์) รอสักครู่แล้วลองใหม่")
            return

    if (sym == SPOT) != (src == SPOT):
        st.warning("แหล่งข้อมูลที่ใช้จริงเป็นคนละสินค้ากับที่เลือก (สปอต/ฟิวเจอร์สราคาต่างกันได้หลายดอลลาร์) "
                   "ให้ใช้ตัวเลข 'ห่างจากราคาตอนนี้' ในตารางแทนราคาเต็ม")

    # รวมราคาทิกสดเข้ากับกราฟ (ใช้ได้เฉพาะเมื่อข้อมูลมาจาก Yahoo สัญลักษณ์เดียวกัน)
    now_ts = pd.Timestamp.now(tz=TZ)
    data_age = now_ts - df.index[-1]
    live_age = None
    lv = LIVE.get(sym)
    if data_ok and src == sym and lv and time.time() - lv["ts"] < 20:
        tick_t = pd.Timestamp(lv["ts"], unit="s", tz="UTC").tz_convert(TZ)
        merged = merge_tick(df, lv["price"], tick_t, P["step"])
        if merged is not None:
            df, live_age = merged, time.time() - lv["ts"]

    stale = live_age is None and data_age > max(4 * P["step"], pd.Timedelta(minutes=5))
    fresh = data_ok and not stale
    if stale:
        mins = int(data_age.total_seconds() // 60)
        st.warning(f"ข้อมูลแท่งล่าสุดเก่า {mins // 60} ชม. {mins % 60} นาที ตลาดอาจปิดอยู่ "
                   "สัญญาณด้านล่างอาจไม่ใช่ปัจจุบัน (ปิดการแจ้งเตือนอัตโนมัติไว้)")

    try:
        A = analyse(df, P, min_score, spread_usd, intrabar)
        bkey = (sym, hold_key, src, str(A["d"].index[-2]), min_score, spread_usd)
        if st.session_state.get("bt_key") != bkey:             # คำนวณใหม่เฉพาะเมื่อมีแท่งปิดใหม่
            st.session_state["bt_key"] = bkey
            st.session_state["bt"] = backtest(A["d"], P, spread_usd)
        bt = st.session_state["bt"]
    except Exception as e:
        st.error(f"วิเคราะห์ข้อมูลไม่สำเร็จ: {e}")
        return

    px = A["price"]
    c1, c2, c3 = st.columns(3)
    prev = st.session_state.get("prev_price")
    c1.metric("ราคาล่าสุด", f"{px:.2f}", None if prev is None else f"{px - prev:+.2f}")
    st.session_state["prev_price"] = px
    c2.metric("ATR (ความผันผวน)", f"{A['atr']:.2f}")
    c3.metric("อัปเดตล่าสุด", datetime.now(ZoneInfo(TZ)).strftime("%H:%M:%S"))

    label = {"up": "ขาขึ้น 📈", "down": "ขาลง 📉", "side": "ไซด์เวย์ / ไม่ชัด ➖"}[A["bias"]]
    regime_txt = {"trend": "ตลาดมีเทรนด์", "range": "ตลาดไซด์เวย์", "mixed": "ภาวะกำกวม (ไม่เทรด)"}[A["regime"]]
    st.subheader(f"แนวโน้ม {tf}: {label}")
    left = A["d"].index[-1] + P["step"] - now_ts
    bar_txt = (f" | ⏳ แท่ง {tf} ปิดใน {fmt_td(left)}" if fresh and pd.Timedelta(0) < left <= P["step"] else "")
    st.caption(f"{hold_key} | {regime_txt} (ADX) | คะแนนขั้นต่ำ {min_score}{bar_txt}")
    if intrabar:
        st.info("⚡ โหมดเรียลไทม์: สัญญาณมาจากแท่งที่ยังวิ่งอยู่ อาจหายก่อนแท่งปิด ใช้เฉพาะเมื่อไหวต่อความเสี่ยงที่สูงกว่า")

    p, status = A["plan"], A["status"]
    ci = -1 if intrabar else -2
    g = bt[p["strat"]]["grade"] if p and p["strat"] in bt else None
    blocked = bool(p) and status == "now" and gate and g != "pass"
    zlo = zhi = None
    if p:
        zlo, zhi = p["lo"], p["hi"]
        if zhi - zlo < 0.3 * A["atr"]:
            m = (zlo + zhi) / 2
            zlo, zhi = m - 0.15 * A["atr"], m + 0.15 * A["atr"]
        zlo, zhi = float(zlo), float(zhi)

    exit_left = None
    if p and p["exit_at"] is not None:
        exit_left = p["exit_at"] - now_ts

    if p:
        buy_ = p["side"] == "buy"
        in_zone = zlo <= px <= zhi
        now_ = status == "now"
        if in_zone:
            state_txt = ("🎯 ราคาอยู่ในโซนเข้าแล้ว" if now_
                         else "🎯 ราคาแตะโซนแล้ว แต่ยังไม่มีแท่งปิดยืนยัน ห้ามเข้าทันที")
        elif px > zhi:
            gap = px - zhi
            state_txt = (f"⏳ รอราคา<b>ลง</b>อีก {gap:.2f} ดอลลาร์ ถึงจะเข้าโซน" if buy_
                         else f"⚠️ ราคาอยู่<b>เหนือ</b>โซน SELL {gap:.2f} ดอลลาร์ (เลยโซนไปแล้ว อย่าไล่เข้า)")
        else:
            gap = zlo - px
            state_txt = (f"⚠️ ราคาอยู่<b>ใต้</b>โซน BUY {gap:.2f} ดอลลาร์ (เลยโซนไปแล้ว อย่าไล่เข้า)" if buy_
                         else f"⏳ รอราคา<b>ขึ้น</b>อีก {gap:.2f} ดอลลาร์ ถึงจะเข้าโซน")
        side_txt = "BUY" if buy_ else "SELL"
        timer_html = ""
        if blocked:
            color, border = "#7a7a7a", "dashed"
            head = f"⛔ มีสัญญาณ {side_txt} แต่กติกานี้ยังไม่ผ่านเกณฑ์ย้อนหลัง · ไม่แนะนำให้เข้า"
            foot = ("ยังไม่มีสถิติย้อนหลังที่พิสูจน์ว่าคุ้มหลังหักสเปรด (ดูกล่อง 'เปรียบเทียบกลยุทธ์') "
                    "ถ้าจะดูโซนต่อ ให้ปิดตัวเลือก 'ซ่อนคำสั่งเข้าไม้...' ที่แถบซ้าย แต่ความเสี่ยงเป็นของคุณ")
        elif now_:
            color = "#00b050" if buy_ else "#e02020"
            head = (("⚡ ชั่วคราว (แท่งยังไม่ปิด) " if A["intrabar"] else "✅ ") + f"เข้า {side_txt} ได้ตอนนี้ · {STRATS[p['strat']]}")
            border = "solid"
            foot = (("⚠️ แท่งยังไม่ปิด สัญญาณนี้อาจหายก่อนปิดแท่ง (ดูสถิติยืนยันในกล่องเรียลไทม์) " if A["intrabar"]
                     else "แท่งสัญญาณปิดยืนยันแล้ว ") + "เข้าในโซนแล้วตั้ง SL ตามด้านบนทันที "
                    f"ถ้าไม่ถึง TP/SL ให้ปิดที่ราคาตลาดเมื่อครบ {P['max_bars']} แท่ง")
            if exit_left is not None:
                timer_html = (f"<span>⏱ ปิดไม้ภายใน <b>{fmt_td(exit_left)}</b> "
                              f"({p['exit_at'].strftime('%H:%M:%S')})</span>")
        else:
            color = "#b8860b"
            head = f"👀 เฝ้ารอ {side_txt} · ยังไม่ใช่สัญญาณเข้า อย่าเพิ่งกดออเดอร์"
            border = "dashed"
            foot = "จะเข้าได้ก็ต่อเมื่อการ์ดนี้เปลี่ยนเป็น '✅ เข้าได้ตอนนี้' (มีแท่งปิดยืนยันครบ)"
        st.markdown(
            f"""<div style="border:3px {border} {color};border-radius:12px;padding:14px 18px;margin:6px 0 12px 0;
            background:{color}18;">
            <div style="font-size:1.05rem;font-weight:700;color:{color};">{head}</div>
            <div style="font-size:2.1rem;font-weight:800;line-height:1.25;">
            {'🟢 BUY' if buy_ else '🔴 SELL'} โซน {zlo:.2f} – {zhi:.2f}</div>
            <div style="font-size:1.05rem;margin:4px 0 8px 0;">{state_txt}</div>
            <div style="display:flex;gap:18px;flex-wrap:wrap;font-size:1rem;">
            <span>🛑 SL <b>{p['sl']:.2f}</b></span>
            <span>🎯 TP1 <b>{p['tp1']:.2f}</b></span>
            <span>🎯 TP2 <b>{p['tp2']:.2f}</b></span>
            <span>RR <b>1 : {p['rr']:.1f}</b> (สุทธิ 1 : {p['rr_net']:.2f})</span>{timer_html}</div>
            <div style="font-size:0.85rem;opacity:0.75;margin-top:6px;">{foot}</div>
            </div>""", unsafe_allow_html=True)
        sc_txt = "ความมั่นใจของสัญญาณ" if now_ else "คะแนนเงื่อนไขตอนนี้ (ถ้าถึงโซนแล้วยืนยันครบ)"
        st.progress(min(1.0, max(0.0, p["score"] / 100)), text=f"{sc_txt}: {p['score']:.0f}/100")
        r_ = bt.get(p["strat"])
        if r_:
            icon = {"pass": "🟢 ผ่านเกณฑ์", "few": "🟡 ข้อมูลย้อนหลังน้อย (ยังสรุปไม่ได้)", "fail": "🔴 ไม่ผ่านเกณฑ์"}[r_["grade"]]
            st.caption(f"ความน่าเชื่อถือจากสถิติย้อนหลัง (หักสเปรดแล้ว): {icon} · {r_['n']} ไม้ ชนะ {r_['winrate']:.0f}% "
                       f"R/ไม้ {r_['exp_r']:+.2f} · ช่วงทดสอบล่าสุด 30%: {r_['oos']['n']} ไม้ R/ไม้ {r_['oos']['exp_r']:+.2f}")
        others = [f"{SHORT[k]} {s.upper()} ({sc:.0f})" for k, s, sc in A["fired"][1:]]
        if others:
            st.caption("กลยุทธ์อื่นที่ให้สัญญาณพร้อมกันที่แท่งนี้: " + ", ".join(others))
    elif status in ("dead", "late"):
        st.warning("⛔ " + A["msg"])
    else:
        st.warning("⛔ ยังไม่ควรเข้าไม้ · " + A["msg"])

    if p:
        def dist(v):
            return f"{v - px:+.2f}"

        exit_txt = (f"{p['exit_at'].strftime('%H:%M:%S')} (อีก {fmt_td(exit_left)})" if exit_left is not None
                    else f"ครบ {P['max_bars']} แท่งหลังเข้า")
        rows = [
            ("ทิศทาง", p["side"].upper(), "-"),
            ("กลยุทธ์", STRATS[p["strat"]], "-"),
            ("โซนเข้า", f"{zlo:.2f} - {zhi:.2f}", dist((zlo + zhi) / 2)),
            ("Stop loss", f"{p['sl']:.2f}", dist(p["sl"])),
            ("TP1", f"{p['tp1']:.2f}", dist(p["tp1"])),
            ("TP2", f"{p['tp2']:.2f}", dist(p["tp2"])),
            ("RR ถึง TP1 (ก่อน / หลังหักสเปรด)", f"1 : {p['rr']:.1f} / 1 : {p['rr_net']:.2f}", "-"),
            ("ระยะ SL / สเปรดกินกี่ % ของ SL", f"{p['sl_dist']:.2f} ดอลลาร์ / {p['sp_pct']:.0f}%", "-"),
            ("ปิดไม้ที่ราคาตลาดเมื่อ", exit_txt, "-"),
            ("ระยะเวลาถือ", P["hold"], "-"),
        ]
        t = pd.DataFrame(rows, columns=["รายการ", "ค่า", "ห่างจากราคาตอนนี้ ($)"])
        stretch(st.dataframe, t, hide_index=True, key="tbl-plan")
        st.caption("ราคา Yahoo อาจต่างจากโบรกเกอร์ใน MT5 ได้ไม่กี่ดอลลาร์ ถ้าไม่ตรง ให้ใช้คอลัมน์ 'ห่างจากราคาตอนนี้' "
                   "นับจากราคาที่เห็นใน MT5 แทนตัวเลขเต็ม")
        for w in A["warns"]:
            st.warning(w)

        if status == "now" and fresh and not blocked and tg_token and tg_chat:
            akey = f"{sym}|{hold_key}|{A['bar']}|{p['side']}"
            al = get_alerted()["keys"]
            if akey not in al:                                   # แจ้งครั้งเดียวต่อแท่ง (กันซ้ำข้ามแท็บ)
                al.append(akey)
                del al[:-200]
                msg = (("⚡ชั่วคราว (แท่งยังไม่ปิด) " if A["intrabar"] else "") + f"สัญญาณ {p['side'].upper()} XAUUSD {tf} ({hold_key})\nกลยุทธ์ {STRATS[p['strat']]} "
                       f"คะแนน {p['score']:.0f}/100\nโซนเข้า {zlo:.2f}-{zhi:.2f} (ราคา Yahoo ตอนนี้ {px:.2f})\n"
                       f"SL {p['sl']:.2f} ({dist(p['sl'])})\nTP1 {p['tp1']:.2f} ({dist(p['tp1'])}) | "
                       f"TP2 {p['tp2']:.2f} ({dist(p['tp2'])})\nปิดที่ราคาตลาดเวลา {p['exit_at'].strftime('%H:%M:%S')} "
                       f"ถ้าไม่ถึง TP/SL | สเปรดกิน {p['sp_pct']:.0f}% ของ SL")
                threading.Thread(target=send_tg, args=(msg, tg_token, tg_chat), daemon=True).start()

    with st.expander("เงื่อนไขที่ใช้ตัดสิน", expanded=False):
        for k, v in A["checks"].items():
            st.write(("✅ " if v else "❌ ") + k)
        st.write(f"แนวรับใกล้สุด: {A['sup']:.2f}" if A["sup"] else "แนวรับใกล้สุด: -")
        st.write(f"แนวต้านใกล้สุด: {A['res']:.2f}" if A["res"] else "แนวต้านใกล้สุด: -")

    with st.expander("เปรียบเทียบกลยุทธ์ + สัญญาณล่าสุด + ผลย้อนหลังคร่าวๆ (หักสเปรดแล้ว)", expanded=False):
        dd = A["d"]
        ens = A["enabled"]
        brow = []
        for k in ens:
            r = bt[k]
            now = "BUY" if dd[f"{k}_buy"].iloc[ci] else "SELL" if dd[f"{k}_sell"].iloc[ci] else "-"
            brow.append({"กลยุทธ์": STRATS[k], "สัญญาณตอนนี้": now, "ไม้ย้อนหลัง": r["n"],
                         "ชนะ %": f"{r['winrate']:.0f}" if r["n"] else "-",
                         "ค่าคาดหวัง R/ไม้": f"{r['exp_r']:+.2f}" if r["n"] else "-",
                         "R/ไม้ 30% ล่าสุด": f"{r['oos']['exp_r']:+.2f}" if r["oos"]["n"] else "-",
                         "เกณฑ์": {"pass": "🟢", "few": "🟡", "fail": "🔴"}[r["grade"]]})
        if len(ens) > 1:
            r = bt["all"]
            brow.append({"กลยุทธ์": "รวมทั้งระบบ (เลือกสัญญาณคะแนนสูงสุด)", "สัญญาณตอนนี้": "",
                         "ไม้ย้อนหลัง": r["n"], "ชนะ %": f"{r['winrate']:.0f}" if r["n"] else "-",
                         "ค่าคาดหวัง R/ไม้": f"{r['exp_r']:+.2f}" if r["n"] else "-",
                         "R/ไม้ 30% ล่าสุด": f"{r['oos']['exp_r']:+.2f}" if r["oos"]["n"] else "-",
                         "เกณฑ์": {"pass": "🟢", "few": "🟡", "fail": "🔴"}[r["grade"]]})
        stretch(st.dataframe, pd.DataFrame(brow), hide_index=True, key="tbl-bt")
        st.caption("ค่าคาดหวัง R/ไม้ > 0 = ในข้อมูลช่วงสั้นที่โหลดมา กลยุทธ์นั้นเฉลี่ยกำไรหลังหักสเปรด "
                   f"${spread_usd:.2f}/ไม้ (1R = เสี่ยงต่อไม้หนึ่งหน่วย) จำลองเข้าที่ราคาปิดแท่งสัญญาณ ออกที่ SL / TP1 "
                   f"หรือปิดที่ราคาตลาดเมื่อครบเวลาถือ ({P['max_bars']} แท่ง) ยังไม่รวมสลิป จำนวนไม้น้อยอาจแกว่งมาก "
                   "เกณฑ์: 🟢 ผ่าน = ไม้ ≥20, R/ไม้ รวม >0.05 และช่วง 30% ล่าสุด (ที่ไม่ได้ใช้ปรับค่า) เป็นบวก / 🟡 ไม้น้อยเกินสรุป / 🔴 ไม่ผ่าน "
                   "ถ้าไม่ใช่ 🟢 อย่าใช้เงินจริง ผลในอดีตไม่รับประกันอนาคต")

        sg = dd[dd.buy_sig | dd.sell_sig].tail(6).iloc[::-1]
        if len(sg):
            def names(row):
                ks = [SHORT[k] for k in ens if row[f"{k}_buy"] or row[f"{k}_sell"]]
                return "+".join(ks)

            stretch(
                st.dataframe,
                pd.DataFrame({
                    "เวลา (ไทย)": [i.strftime("%d/%m %H:%M") for i in sg.index],
                    "ทิศ": ["BUY" if b else "SELL" for b in sg.buy_sig],
                    "กลยุทธ์": [names(r_) for _, r_ in sg.iterrows()],
                    "คะแนน": [f"{(b if bs_ else s):.0f}" for b, s, bs_ in zip(sg.buy_sc, sg.sell_sc, sg.buy_sig)],
                    "ราคาปิดแท่งนั้น": [f"{x:.2f}" for x in sg.close],
                }),
                hide_index=True,
                key="tbl-sig",
            )
        else:
            st.write("ยังไม่มีสัญญาณในช่วงข้อมูลที่โหลด")

    if intrabar:
        log = get_iblog().setdefault((sym, hold_key), [])
        if A["fired"] and not any(x["bar"] == A["bar"] for x in log):
            k_, s_, _sc = A["fired"][0]
            log.append(dict(bar=A["bar"], side=s_, strat=k_, ref=px, ok=None))
        for x in log:                                          # แท่งที่ปิดไปแล้ว: ตรวจว่าสัญญาณยังอยู่ไหม
            if x["ok"] is None and x["bar"] < A["bar"]:
                try:
                    x["ok"] = bool(A["d"].loc[x["bar"], f"{x['strat']}_{x['side']}"])
                except Exception:
                    x["ok"] = None
        del log[:-60]
        with st.expander("สถิติสัญญาณเรียลไทม์: ยืนยันเมื่อแท่งปิดหรือหายไป", expanded=False):
            done = [x for x in log if x["ok"] is not None]
            if done:
                okn = sum(1 for x in done if x["ok"])
                st.write(f"สัญญาณที่เห็นก่อนแท่งปิด {len(done)} ครั้ง ยืนยันเมื่อปิดแท่ง {okn} ครั้ง "
                         f"({100 * okn / len(done):.0f}%) ที่เหลือหายไปก่อนปิด ยิ่งอัตราต่ำ ยิ่งไม่ควรเข้าก่อนแท่งปิด")
            else:
                st.write("ยังไม่มีข้อมูล รอให้มีสัญญาณเรียลไทม์และแท่งปิดสักครั้ง (สถิตินี้รีเซ็ตเมื่อแอปรีสตาร์ท)")

    # กราฟ (ตัด timezone ออกเพื่อให้ Plotly แสดงเวลาไทยตรงๆ)
    d = A["d"].tail(120).copy()
    d.index = d.index.tz_localize(None)
    x_end = d.index[-1] + P["step"] * 14
    fig = go.Figure(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close,
                                   name="ราคา", increasing_line_color="#1b7a62", decreasing_line_color="#b3372f"))
    for col_, nm in (("ema_t", "EMA200"), ("ema_s", "EMA50"), ("ema_f", "EMA21")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col_], name=nm, line=dict(width=1.2)))
    if p and p["strat"] == "reversal":
        for col_ in ("bb_u", "bb_l"):
            fig.add_trace(go.Scatter(x=d.index, y=d[col_], name="Bollinger", legendgroup="bb",
                                     showlegend=(col_ == "bb_u"), line=dict(width=1, dash="dot", color="#8a8a8a")))

    bsig, ssig = d[d.buy_sig], d[d.sell_sig]
    if len(bsig):
        fig.add_trace(go.Scatter(x=bsig.index, y=bsig.low - 0.5 * bsig.atr, mode="markers", name="สัญญาณ BUY",
                                 marker=dict(symbol="triangle-up", size=17, color="#00b050",
                                             line=dict(width=1, color="#005a28")),
                                 text=[f"{x:.0f}" for x in bsig.buy_sc],
                                 hovertemplate="BUY คะแนน %{text}<extra></extra>"))
    if len(ssig):
        fig.add_trace(go.Scatter(x=ssig.index, y=ssig.high + 0.5 * ssig.atr, mode="markers", name="สัญญาณ SELL",
                                 marker=dict(symbol="triangle-down", size=17, color="#e02020",
                                             line=dict(width=1, color="#7a0000")),
                                 text=[f"{x:.0f}" for x in ssig.sell_sc],
                                 hovertemplate="SELL คะแนน %{text}<extra></extra>"))

    # กรอบโซนเข้า + แถบเป้ากำไร + เส้น SL/TP + เส้นเวลาปิดไม้
    if p:
        buy = p["side"] == "buy"
        now_ = status == "now"
        if now_:
            col, fill = ("#00b050", "rgba(0,176,80,0.30)") if buy else ("#e02020", "rgba(224,32,32,0.30)")
        else:
            col, fill = "#b8860b", "rgba(184,134,11,0.22)"
        tint = "rgba(0,176,80,0.08)" if buy else "rgba(224,32,32,0.08)"
        x0 = d.index[-30]
        fig.add_shape(type="line", x0=x0, x1=x_end, y0=px, y1=px, line=dict(color="#1f6feb", width=1.5))
        fig.add_annotation(x=d.index[0], y=px, xanchor="left", yanchor="bottom", showarrow=False,
                           text=f"ราคา {px:.2f}", font=dict(color="#1f6feb", size=12))
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=min(p["mid"], p["tp1"]), y1=max(p["mid"], p["tp1"]),
                      fillcolor=tint, line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=zlo, y1=zhi, fillcolor=fill,
                      line=dict(color=col, width=2.5, dash="solid" if now_ else "dash"))
        tag = ("เข้า " if now_ else "เฝ้ารอ ") + ("BUY" if buy else "SELL")
        fig.add_annotation(x=x_end, y=(zlo + zhi) / 2, xanchor="right", showarrow=False,
                           font=dict(color=col, size=15), text=f"<b>{tag}</b> {zlo:.2f}-{zhi:.2f}",
                           bgcolor="rgba(255,255,255,0.92)", bordercolor=col, borderwidth=2)
        for y, nm, c2_ in ((p["sl"], "SL", "#b3372f"), (p["tp1"], "TP1", "#a87a1f"), (p["tp2"], "TP2", "#a87a1f")):
            fig.add_shape(type="line", x0=x0, x1=x_end, y0=y, y1=y, line=dict(color=c2_, width=1.5, dash="dot"))
            fig.add_annotation(x=x_end, y=y, xanchor="right", yanchor="bottom", showarrow=False,
                               text=f"{nm} {y:.1f}", font=dict(color=c2_, size=12))
        if p["exit_at"] is not None:
            ex = p["exit_at"].tz_localize(None) if p["exit_at"].tzinfo is not None else p["exit_at"]
            if ex <= x_end:                       # เส้นประแนวตั้ง = เวลาปิดไม้ที่ราคาตลาด
                fig.add_shape(type="line", x0=ex, x1=ex, y0=0, y1=1, yref="paper",
                              line=dict(color="#6b6b6b", width=1.5, dash="dash"))
                fig.add_annotation(x=ex, y=1, yref="paper", yanchor="bottom", showarrow=False,
                                   text="ปิดไม้", font=dict(color="#6b6b6b", size=12))

    rev = f"{sym}-{hold_key}"           # เปลี่ยนเฉพาะตอนสลับสินค้า/ระยะเวลาถือไม้
    fig.update_xaxes(range=[d.index[0], x_end], uirevision=rev)
    fig.update_yaxes(uirevision=rev)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_rangeslider_visible=False,
                      uirevision=rev, transition=dict(duration=0))

    stretch(chart_box.plotly_chart, fig, config=dict(displaylogo=False, scrollZoom=True),
            key=f"chart-{sym}-{hold_key}")
    st.caption(src_note + (f" | 🟢 ทิกสด WebSocket (ล่าสุด {live_age:.0f} วินาทีที่แล้ว)" if live_age is not None
                           else " | 🟡 ยังไม่มีทิกสด ใช้ข้อมูลที่ดึงเป็นรอบ (ตลาดอาจปิด หรือ WebSocket ยังเชื่อมไม่ติด)"
                           + (f" [WebSocket: {WS['err']}]" if WS.get("err") else "")))
    st.caption(f"กราฟอัปเดตอัตโนมัติทุก {speed} วินาที | ราคาจาก Yahoo Finance ไม่ใช่ทิกสดของโบรกเกอร์ "
               "อาจช้ากว่าและกระโดดเป็นช่วงๆ")


live_view()
