"""
Gold Analyzer - แอปวิเคราะห์ราคาทอง XAUUSD แบบอัตโนมัติ (ไม่ต้องใช้ MT5)
ดึงราคาล่าสุดจาก Yahoo Finance + ทิกสด WebSocket แล้วบอกแนวโน้ม + จุดเข้า Buy/Sell + SL/TP
รันด้วย: streamlit run app.py   (หรือดับเบิลคลิก run.bat)
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
         "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}


def clean(df):
    """ทำความสะอาด OHLC + แปลงเวลาเป็นเวลาไทย (tz-aware)"""
    df = df.rename(columns=str.lower)[BASE_COLS].astype(float).dropna()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.high >= df.low)]


def merge_tick(df, price, tick_time, step, max_gap=3):
    """รวมราคาทิกสดเข้ากับกราฟ: ถ้ายังอยู่ในแท่งล่าสุดก็อัปเดตแท่งนั้น
    ถ้าข้ามไปแท่งใหม่แล้ว (Yahoo ยังไม่ส่งแท่งใหม่) ก็สร้างแท่งใหม่ชั่วคราว
    คืน None ถ้าข้อมูลที่โหลดมาเก่าเกินไปเมื่อเทียบกับทิก (ไม่ควรเอามาต่อกัน)"""
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


def add_ind(df):
    c = df["close"]
    df["ema_t"] = c.ewm(span=200, adjust=False).mean()
    df["ema_s"] = c.ewm(span=50, adjust=False).mean()
    df["ema_f"] = c.ewm(span=21, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # ถ้าไม่มีแรงขายเลย RSI ต้องเป็น 100 (ไม่ใช่ NaN) / ถ้านิ่งสนิทให้เป็น 50
    df["rsi"] = rsi.where(dn != 0, np.where(up > 0, 100.0, 50.0))
    pc = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    # สัญญาณย้อนหลังทุกแท่ง (เงื่อนไขชุดเดียวกับสัญญาณปัจจุบัน = แหล่งความจริงเดียว)
    up_t = (df.close > df.ema_t) & (df.ema_t > df.ema_t.shift(5)) & (df.ema_f > df.ema_s)
    dn_t = (df.close < df.ema_t) & (df.ema_t < df.ema_t.shift(5)) & (df.ema_f < df.ema_s)
    df["up_t"], df["dn_t"] = up_t, dn_t
    df["buy_sig"] = up_t & (df.low <= df.ema_f) & (df.close > df.ema_f) & df.rsi.between(45, 68)
    df["sell_sig"] = dn_t & (df.high >= df.ema_f) & (df.close < df.ema_f) & df.rsi.between(32, 55)
    df.iloc[-1, df.columns.get_loc("buy_sig")] = False    # แท่งสุดท้ายยังไม่ปิด ไม่นับ
    df.iloc[-1, df.columns.get_loc("sell_sig")] = False
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


def analyse(df, prof, off=0.0):
    d = add_ind(df.copy())
    b1, b6 = d.iloc[-2], d.iloc[-7]          # แท่งที่ปิดแล้ว
    price = float(d.iloc[-1]["close"])
    atr = float(b1["atr"])
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("ค่า ATR ใช้ไม่ได้ (ข้อมูลนิ่งหรือไม่พอ)")
    up = bool(b1.up_t)
    dn = bool(b1.dn_t)
    bias = "up" if up else "down" if dn else "side"

    buy_trig = bool(d["buy_sig"].iloc[-2])
    sell_trig = bool(d["sell_sig"].iloc[-2])

    checks = {
        "ราคาเทียบ EMA200 และ EMA200 ชี้ทิศทางเดียวกัน": up or dn,
        "EMA21 / EMA50 เรียงตามเทรนด์": (b1.ema_f > b1.ema_s) if bias == "up" else (b1.ema_f < b1.ema_s) if bias == "down" else False,
        "ราคาย่อแตะ EMA21 แล้วปิดกลับตามเทรนด์": buy_trig or sell_trig or
            (bias == "up" and b1.low <= b1.ema_f and b1.close > b1.ema_f) or
            (bias == "down" and b1.high >= b1.ema_f and b1.close < b1.ema_f),
        f"RSI อยู่ในช่วงเหมาะสม (ตอนนี้ {b1.rsi:.0f})": (45 <= b1.rsi <= 68) if bias == "up" else (32 <= b1.rsi <= 55) if bias == "down" else False,
    }
    checks = {k: bool(v) for k, v in checks.items()}

    sup, res = swing_levels(d)
    near_res = min([x for x in res if x > price], default=None)
    near_sup = max([x for x in sup if x < price], default=None)

    side = "buy" if bias == "up" else "sell" if bias == "down" else None
    plan, status, warns = None, "no", []
    if side:
        if buy_trig or sell_trig:
            status = "now"
            lo = hi = price
        else:
            status = "wait"
            lo, hi = (b1.ema_f - 0.3 * atr, b1.ema_f + 0.1 * atr) if side == "buy" else (b1.ema_f - 0.1 * atr, b1.ema_f + 0.3 * atr)
        mid = (lo + hi) / 2
        s = 1 if side == "buy" else -1
        sl = mid - s * prof["sl"] * atr
        tp1 = mid + s * prof["tp"] * atr
        tp2 = mid + s * prof["tp"] * 1.6 * atr
        plan = dict(side=side, lo=lo, hi=hi, mid=mid, sl=sl, tp1=tp1, tp2=tp2, sl_dist=abs(mid - sl),
                    rr=abs(tp1 - mid) / abs(mid - sl))
        if side == "buy" and near_res and near_res < tp1:
            warns.append(f"มีแนวต้านที่ {near_res + off:.2f} ก่อนถึง TP1 ราคาอาจชนแนวนี้ก่อน")
        if side == "sell" and near_sup and near_sup > tp1:
            warns.append(f"มีแนวรับที่ {near_sup + off:.2f} ก่อนถึง TP1 ราคาอาจชนแนวนี้ก่อน")
        if side == "buy" and b1.rsi > 70:
            warns.append("RSI สูง ระวังไล่ราคา")
        if side == "sell" and b1.rsi < 30:
            warns.append("RSI ต่ำ ระวังไล่ราคา")
        if abs(price - b1.ema_f) > 2.5 * atr and status == "wait":
            warns.append("ราคาอยู่ห่างจากจุดย่อมาก อย่าไล่เข้า รอให้ย้อนกลับมาที่โซน")
    return dict(d=d, bias=bias, status=status, plan=plan, price=price, atr=atr, checks=checks,
                sup=near_sup, res=near_res, warns=warns, bar=d.index[-2])


def backtest(d, sl_m, tp_m, max_bars=60):
    """จำลองสัญญาณย้อนหลังในข้อมูลที่โหลดมา: เข้าที่ราคาปิดแท่งสัญญาณ, SL/TP1 ตามตัวคูณ ATR
    - ถ้าแท่งเดียวชนทั้ง SL และ TP นับเป็นแพ้ (สมมติแบบระมัดระวัง)
    - ไม่เปิดไม้ซ้อน: ข้ามสัญญาณที่เกิดระหว่างไม้เดิมยังเปิดอยู่
    - ยังไม่รวมสเปรด/สลิป จึงเป็นภาพคร่าวๆ เท่านั้น"""
    hi, lo, cl, atr = (d[c].to_numpy(float) for c in ("high", "low", "close", "atr"))
    buy = d["buy_sig"].to_numpy(bool)
    n, wins, losses, free_from = len(d), 0, 0, 0
    for i in np.flatnonzero(buy | d["sell_sig"].to_numpy(bool)):
        if i < free_from or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        s = 1 if buy[i] else -1
        sl, tp = cl[i] - s * sl_m * atr[i], cl[i] + s * tp_m * atr[i]
        for j in range(i + 1, min(n, i + 1 + max_bars)):
            hit_sl = lo[j] <= sl if s == 1 else hi[j] >= sl
            hit_tp = hi[j] >= tp if s == 1 else lo[j] <= tp
            if hit_sl or hit_tp:                 # ชนทั้งคู่ในแท่งเดียว => แพ้
                wins, losses = (wins, losses + 1) if hit_sl else (wins + 1, losses)
                free_from = j + 1
                break
    total = wins + losses
    rr = tp_m / sl_m
    return dict(n=total, wins=wins, losses=losses,
                winrate=(100 * wins / total) if total else 0.0,
                exp_r=((wins * rr - losses) / total) if total else 0.0)


def lot_for(sl_dist, balance, risk_pct, contract, min_lot):
    risk_money = balance * risk_pct / 100
    per_lot = sl_dist * contract
    if per_lot <= 0 or min_lot <= 0:
        return 0.0, "คำนวณ lot ไม่ได้"
    lot = np.floor(round(risk_money / per_lot / min_lot, 6)) * min_lot    # round กันเศษทศนิยมลอยตัว
    if lot < min_lot:
        actual = min_lot * per_lot
        if actual > risk_money * 2:
            return 0.0, f"lot ขั้นต่ำเสี่ยง ${actual:.2f} เกินกรอบ ${risk_money:.2f} มาก ไม่ควรเข้าไม้นี้ (ลองบัญชี Cent/Micro หรือเพิ่มทุน)"
        return min_lot, f"ใช้ lot ขั้นต่ำ เสี่ยง ${actual:.2f}"
    return round(float(lot), 2), f"เสี่ยงประมาณ ${lot * per_lot:.2f}"
# ===================================================================================
# <<< CORE END
# ===================================================================================

# ค่าตามแต่ละ Timeframe: interval/period ของ Yahoo, ตัวคูณ ATR สำหรับ SL/TP
PROFILES = {
    "M1":  dict(interval="1m",  period="5d",   sl=1.5, tp=2.25),
    "M5":  dict(interval="5m",  period="30d",  sl=1.5, tp=2.5),
    "M15": dict(interval="15m", period="30d",  sl=1.5, tp=3.0),
    "H1":  dict(interval="1h",  period="180d", sl=1.5, tp=3.0),
}
SYMBOLS = {"XAUUSD=X (ทองสปอต)": "XAUUSD=X", "GC=F (ทองฟิวเจอร์ส COMEX)": "GC=F"}
SPOT = "XAUUSD=X"
TD_INTERVAL = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}


def _secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("ตั้งค่า")
sym_label = sb.selectbox("แหล่งราคา", list(SYMBOLS))
tf = sb.selectbox("Timeframe", list(PROFILES), index=1)
offset = sb.number_input("ส่วนต่างราคาโบรกเกอร์ (โบรกเกอร์ - Yahoo)", value=0.0, step=0.1,
                         help="เทียบราคาในแอปกับ MT5 แล้วใส่ผลต่างตรงนี้ เพื่อให้จุดเข้าตรงกับกราฟที่คุณเทรดจริง")
sb.subheader("จัดการความเสี่ยง")
balance = sb.number_input("ยอดเงิน (USD)", value=100.0, min_value=1.0, step=10.0)
risk_pct = sb.number_input("เสี่ยงต่อไม้ (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
contract = sb.number_input("Contract size (ออนซ์/lot)", value=100.0, min_value=1.0, step=1.0)
min_lot = sb.number_input("lot ขั้นต่ำ", value=0.01, min_value=0.01, step=0.01, format="%.2f")
sb.subheader("แจ้งเตือน Telegram (ไม่บังคับ)")
tg_token = sb.text_input("Bot token", value=_secret("TG_TOKEN"), type="password")
tg_chat = sb.text_input("Chat ID", value=str(_secret("TG_CHAT")))
sb.subheader("แหล่งราคาสำรอง (ไม่บังคับ)")
td_key = sb.text_input("Twelve Data API key", value=_secret("TWELVE_KEY"), type="password",
                       help="สมัครฟรีที่ twelvedata.com ใช้เมื่อ Yahoo ไม่ส่งข้อมูล (เซิร์ฟเวอร์คลาวด์มักถูก Yahoo จำกัด) "
                            "โควตาฟรีจำกัด แอปจึงดึงจากแหล่งนี้ไม่เกินทุก ~2 นาที")

speed = sb.selectbox("ความเร็วอัปเดตกราฟ (วินาที)", [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60], index=1,
                     help="เร็วสุดที่ Streamlit วาดกราฟทั้งใบใหม่ได้จริงคือราว 0.1 วินาที (0.01 วินาที = 100 ครั้ง/วินาที ทำไม่ได้ "
                          "และ Yahoo ส่งทิกมาราว 1 ครั้ง/วินาที จึงไม่มีราคาใหม่ให้วาดถี่ขนาดนั้น) "
                          "ถ้าเครื่องหรือเน็ตช้าให้เลือก 0.5-1 วินาที ข้อมูลแท่งเทียนย้อนหลังดึงใหม่ทุก ~5 วินาที")
P = dict(PROFILES[tf], step=STEPS[PROFILES[tf]["interval"]])


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


@st.cache_data(ttl=120, show_spinner=False)
def load_td(interval, key):
    """Twelve Data (สปอต XAU/USD) แคช 2 นาที เพื่อไม่ให้โควตาฟรีหมด"""
    try:
        r = requests.get("https://api.twelvedata.com/time_series", timeout=15, params=dict(
            symbol="XAU/USD", interval=TD_INTERVAL[interval], outputsize=600, timezone="UTC", apikey=key)).json()
        if "values" not in r:
            return None, f"Twelve Data: {r.get('message', 'ไม่มีข้อมูล')}"
        df = pd.DataFrame(r["values"])
        df.index = pd.to_datetime(df.pop("datetime"))
        df = clean(df)
        if len(df) >= 260:
            return df, ""
        return None, f"Twelve Data: ได้ {len(df)} แท่ง"
    except Exception as e:
        return None, f"Twelve Data: {type(e).__name__}"


def load(symbol, interval, period, key):
    """ลองตามลำดับ: แหล่งที่เลือก -> (Twelve Data ถ้าเลือกสปอต) -> Yahoo อีกตัว -> (Twelve Data)
    คืน (df, ข้อความ, รหัสแหล่งที่ใช้จริง) รหัส = สัญลักษณ์ Yahoo หรือ "TD" (สปอต)"""
    other = "GC=F" if symbol == SPOT else SPOT
    order = [("y", symbol)]
    if key and symbol == SPOT:
        order.append(("td", "TD"))
    order.append(("y", other))
    if key and symbol != SPOT:
        order.append(("td", "TD"))
    notes = []
    for kind, code in order:
        df, note = load_yahoo(code, interval, period) if kind == "y" else load_td(interval, key)
        if df is not None:
            name = f"Yahoo ({code})" if kind == "y" else "Twelve Data (XAU/USD)"
            return df, f"ใช้ข้อมูล {name}" + ("" if code == symbol else " แทนแหล่งที่เลือก"), code
        notes.append(note)
    return None, " | ".join(n for n in notes if n), None


LIVE = {}   # symbol -> {"price": ราคาล่าสุด, "ts": เวลาที่รับ}


@st.cache_resource
def start_stream(symbol):
    """เปิด WebSocket ของ Yahoo ไว้เบื้องหลัง รับราคาทิกสดเก็บไว้ใน LIVE (เปิดครั้งเดียวต่อสัญลักษณ์)"""
    if not hasattr(yf, "WebSocket"):
        return None

    def handler(msg):
        try:
            if msg.get("id") not in (None, symbol):
                return
            price = msg.get("price")
            if price is not None:
                LIVE[symbol] = {"price": float(price), "ts": time.time()}
        except Exception:
            pass

    def run():
        while True:
            try:
                with yf.WebSocket() as ws:
                    ws.subscribe([symbol])
                    ws.listen(handler)
            except Exception:
                pass
            time.sleep(5)   # หลุดแล้วต่อใหม่

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ---------------- UI helpers ----------------
_NEW_ST = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


def stretch(fn, *args, **kw):
    """เรียก st.dataframe / st.plotly_chart ให้เต็มความกว้าง รองรับทั้ง Streamlit เก่าและใหม่
    (use_container_width ถูกเลิกใช้ในเวอร์ชันใหม่)"""
    if _NEW_ST:
        try:
            return fn(*args, width="stretch", **kw)
        except Exception:
            pass
    return fn(*args, use_container_width=True, **kw)


# ---------------- UI ----------------
st.title("🪙 Gold Analyzer")
st.caption("วิเคราะห์ XAUUSD อัตโนมัติจากราคาล่าสุด ไม่ใช่การรับประกันผล ตรวจกราฟจริงและตั้ง SL ทุกไม้")


@st.fragment(run_every=speed)
def live_view():
    sym = SYMBOLS[sym_label]
    start_stream(sym)
    key = (sym, tf)
    df, src_note, src = load(sym, P["interval"], P["period"], td_key)
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
            st.info("ถ้าเห็นว่า Yahoo ได้ 0 แท่งทุกตัว แปลว่า Yahoo ปิดกั้นเซิร์ฟเวอร์คลาวด์ ให้ใส่ Twelve Data API key (ฟรี) "
                    "ที่แถบข้างซ้ายหัวข้อ 'แหล่งราคาสำรอง' หรือรอสักครู่แล้วลองใหม่")
            return

    # ถ้าแหล่งที่ใช้จริงเป็นสินค้าคนละประเภท (สปอต vs ฟิวเจอร์ส) ราคาจะต่างกันหลายดอลลาร์
    if (sym == SPOT) != (src in (SPOT, "TD")):
        st.warning("แหล่งข้อมูลที่ใช้จริงเป็นคนละสินค้ากับที่เลือก (สปอต/ฟิวเจอร์สราคาต่างกันได้หลายดอลลาร์) "
                   "ตรวจช่อง 'ส่วนต่างราคาโบรกเกอร์' ให้ตรงกับกราฟที่เทรดจริง")

    # รวมราคาทิกสดเข้ากับกราฟ เพื่อให้ขยับทุกวินาที (ใช้ได้เฉพาะเมื่อข้อมูลมาจาก Yahoo สัญลักษณ์เดียวกัน)
    now_ts = pd.Timestamp.now(tz=TZ)
    data_age = now_ts - df.index[-1]
    live_age = None
    lv = LIVE.get(sym)
    if data_ok and src == sym and lv and time.time() - lv["ts"] < 20:
        tick_t = pd.Timestamp(lv["ts"], unit="s", tz="UTC").tz_convert(TZ)
        merged = merge_tick(df, lv["price"], tick_t, P["step"])
        if merged is not None:
            df, live_age = merged, time.time() - lv["ts"]

    # ข้อมูลเก่าเกินไป (ตลาดปิด/พักเที่ยง/ดึงไม่ได้) => ห้ามถือว่าสัญญาณเป็นปัจจุบัน
    stale = live_age is None and data_age > max(4 * P["step"], pd.Timedelta(minutes=5))
    fresh = data_ok and not stale
    if stale:
        mins = int(data_age.total_seconds() // 60)
        st.warning(f"ข้อมูลแท่งล่าสุดเก่า {mins // 60} ชม. {mins % 60} นาที ตลาดอาจปิดอยู่ "
                   "สัญญาณด้านล่างอาจไม่ใช่ปัจจุบัน (ปิดการแจ้งเตือนอัตโนมัติไว้)")

    try:
        A = analyse(df, P, offset)
    except Exception as e:
        st.error(f"วิเคราะห์ข้อมูลไม่สำเร็จ: {e}")
        return

    o = offset
    c1, c2, c3 = st.columns(3)
    prev = st.session_state.get("prev_price")
    c1.metric("ราคาล่าสุด", f"{A['price'] + o:.2f}", None if prev is None else f"{A['price'] - prev:+.2f}")
    st.session_state["prev_price"] = A["price"]
    c2.metric("ATR (ความผันผวน)", f"{A['atr']:.2f}")
    c3.metric("อัปเดตล่าสุด", datetime.now(ZoneInfo(TZ)).strftime("%H:%M:%S"))

    label = {"up": "ขาขึ้น 📈", "down": "ขาลง 📉", "side": "ไซด์เวย์ / ไม่ชัด ➖"}[A["bias"]]
    st.subheader(f"แนวโน้ม {tf}: {label}")

    p = A["plan"]
    zlo = zhi = None
    if p:
        zlo, zhi = p["lo"], p["hi"]
        if zhi - zlo < 0.3 * A["atr"]:                     # โซนบางเกินไปให้ขยายให้มองเห็น
            m = (zlo + zhi) / 2
            zlo, zhi = m - 0.15 * A["atr"], m + 0.15 * A["atr"]
        zlo, zhi = float(zlo), float(zhi)

    if p:
        buy_ = p["side"] == "buy"
        px = A["price"]
        in_zone = zlo <= px <= zhi
        if in_zone:
            state_txt = "🎯 ราคาอยู่ในโซนเข้าแล้ว" if A["status"] == "now" else "🎯 ราคาแตะโซนแล้ว (รอแท่งปิดยืนยันสัญญาณ)"
        elif px > zhi:
            gap = px - zhi
            state_txt = (f"⏳ รอราคา<b>ลง</b>อีก {gap:.2f} ดอลลาร์ ถึงจะเข้าโซน" if buy_
                         else f"⚠️ ราคาอยู่<b>เหนือ</b>โซน SELL {gap:.2f} ดอลลาร์ (เลยโซนไปแล้ว อย่าไล่เข้า)")
        else:
            gap = zlo - px
            state_txt = (f"⚠️ ราคาอยู่<b>ใต้</b>โซน BUY {gap:.2f} ดอลลาร์ (เลยโซนไปแล้ว อย่าไล่เข้า)" if buy_
                         else f"⏳ รอราคา<b>ขึ้น</b>อีก {gap:.2f} ดอลลาร์ ถึงจะเข้าโซน")
        color = "#00b050" if buy_ else "#e02020"
        head = {"now": f"✅ สัญญาณเข้า {p['side'].upper()} ตอนนี้",
                "wait": f"โซนรอเข้า {p['side'].upper()}"}[A["status"]]
        st.markdown(
            f"""<div style="border:3px solid {color};border-radius:12px;padding:14px 18px;margin:6px 0 12px 0;
            background:{color}18;">
            <div style="font-size:1.05rem;font-weight:700;color:{color};">{head}</div>
            <div style="font-size:2.1rem;font-weight:800;line-height:1.25;">
            {'🟢 BUY' if buy_ else '🔴 SELL'} โซนเข้า {zlo + o:.2f} – {zhi + o:.2f}</div>
            <div style="font-size:1.05rem;margin:4px 0 8px 0;">{state_txt}</div>
            <div style="display:flex;gap:18px;flex-wrap:wrap;font-size:1rem;">
            <span>🛑 SL <b>{p['sl'] + o:.2f}</b></span>
            <span>🎯 TP1 <b>{p['tp1'] + o:.2f}</b></span>
            <span>🎯 TP2 <b>{p['tp2'] + o:.2f}</b></span>
            <span>RR <b>1 : {p['rr']:.1f}</b></span></div>
            <div style="font-size:0.85rem;opacity:0.75;margin-top:6px;">
            ขาเข้าจริงให้รอให้ราคาเข้ากรอบโซนก่อน แล้วค่อยตั้ง SL ตามด้านบน</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("⛔ ยังไม่ควรเข้าไม้ แนวโน้มไม่ชัด รอให้เทรนด์ชัดก่อน")

    if p:
        lot, lot_note = lot_for(p["sl_dist"], balance, risk_pct, contract, min_lot)
        t = pd.DataFrame({
            "รายการ": ["ทิศทาง", "โซนเข้า", "Stop loss", "TP1", "TP2", "RR ถึง TP1", "ระยะ SL", "lot ที่แนะนำ"],
            "ค่า": [p["side"].upper(),
                    f"{zlo + o:.2f} - {zhi + o:.2f}",
                    f"{p['sl'] + o:.2f}", f"{p['tp1'] + o:.2f}", f"{p['tp2'] + o:.2f}",
                    f"1 : {p['rr']:.1f}", f"{p['sl_dist']:.2f} ดอลลาร์",
                    f"{lot:.2f}  ({lot_note})" if lot else "ไม่เปิด: " + lot_note],
        })
        stretch(st.dataframe, t, hide_index=True)
        for w in A["warns"]:
            st.warning(w)

        if A["status"] == "now" and fresh and tg_token and tg_chat:
            akey = f"{sym}|{tf}|{A['bar']}"
            if st.session_state.get("last_alert") != akey:      # แจ้งครั้งเดียวต่อแท่ง
                st.session_state["last_alert"] = akey
                msg = (f"สัญญาณ {p['side'].upper()} XAUUSD {tf}\nเข้า ~{p['mid'] + o:.2f}\nSL {p['sl'] + o:.2f}\n"
                       f"TP1 {p['tp1'] + o:.2f} | TP2 {p['tp2'] + o:.2f}\nlot {lot:.2f}")
                threading.Thread(target=send_tg, args=(msg, tg_token, tg_chat), daemon=True).start()

    with st.expander("เงื่อนไขที่ใช้ตัดสิน", expanded=False):
        for k, v in A["checks"].items():
            st.write(("✅ " if v else "❌ ") + k)
        st.write(f"แนวรับใกล้สุด: {A['sup'] + o:.2f}" if A["sup"] else "แนวรับใกล้สุด: -")
        st.write(f"แนวต้านใกล้สุด: {A['res'] + o:.2f}" if A["res"] else "แนวต้านใกล้สุด: -")

    with st.expander("สัญญาณล่าสุด + ผลย้อนหลังคร่าวๆ", expanded=False):
        bkey = (sym, tf, src, str(A["bar"]))
        if st.session_state.get("bt_key") != bkey:             # คำนวณใหม่เฉพาะเมื่อมีแท่งปิดใหม่
            st.session_state["bt_key"] = bkey
            st.session_state["bt"] = backtest(A["d"], P["sl"], P["tp"])
        bt = st.session_state["bt"]
        if bt["n"]:
            st.write(f"ย้อนหลังในข้อมูลที่โหลด ({bt['n']} ไม้ที่จบแล้ว): ชนะ {bt['wins']} แพ้ {bt['losses']} "
                     f"→ ชนะ {bt['winrate']:.0f}% | ค่าคาดหวัง {bt['exp_r']:+.2f}R ต่อไม้")
            st.caption("จำลองเข้าที่ราคาปิดแท่งสัญญาณ ออกที่ SL/TP1 ยังไม่รวมสเปรด/สลิป และข้อมูลช่วงสั้น "
                       "ใช้ดูภาพรวมเท่านั้น ผลในอดีตไม่รับประกันอนาคต")
        else:
            st.write("ยังไม่มีสัญญาณย้อนหลังพอจะสรุป")
        sg = A["d"][A["d"].buy_sig | A["d"].sell_sig].tail(5).iloc[::-1]
        if len(sg):
            stretch(st.dataframe, pd.DataFrame({
                "เวลา (ไทย)": [i.strftime("%d/%m %H:%M") for i in sg.index],
                "ทิศ": ["BUY" if b else "SELL" for b in sg.buy_sig],
                "ราคาปิดแท่งนั้น": [f"{x + o:.2f}" for x in sg.close]}), hide_index=True)

    # กราฟ (ตัด timezone ออกเพื่อให้ Plotly แสดงเวลาไทยตรงๆ)
    d = A["d"].tail(120).copy()
    d.index = d.index.tz_localize(None)
    x_end = d.index[-1] + P["step"] * 14
    fig = go.Figure(go.Candlestick(x=d.index, open=d.open + o, high=d.high + o, low=d.low + o, close=d.close + o,
                                   name="ราคา", increasing_line_color="#1b7a62", decreasing_line_color="#b3372f"))
    for col, nm in (("ema_t", "EMA200"), ("ema_s", "EMA50"), ("ema_f", "EMA21")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col] + o, name=nm, line=dict(width=1.2)))

    # ลูกศรสัญญาณ: เขียวชี้ขึ้น = BUY (ใต้แท่ง), แดงชี้ลง = SELL (เหนือแท่ง)
    bs, ss = d[d.buy_sig], d[d.sell_sig]
    if len(bs):
        fig.add_trace(go.Scatter(x=bs.index, y=bs.low + o - 0.5 * bs.atr, mode="markers", name="สัญญาณ BUY",
                                 marker=dict(symbol="triangle-up", size=17, color="#00b050", line=dict(width=1, color="#005a28")),
                                 hovertemplate="BUY ที่แท่งนี้<extra></extra>"))
    if len(ss):
        fig.add_trace(go.Scatter(x=ss.index, y=ss.high + o + 0.5 * ss.atr, mode="markers", name="สัญญาณ SELL",
                                 marker=dict(symbol="triangle-down", size=17, color="#e02020", line=dict(width=1, color="#7a0000")),
                                 hovertemplate="SELL ที่แท่งนี้<extra></extra>"))

    # กรอบโซนเข้า + แถบเป้ากำไร + เส้น SL/TP
    if p:
        buy = p["side"] == "buy"
        col, fill = ("#00b050", "rgba(0,176,80,0.30)") if buy else ("#e02020", "rgba(224,32,32,0.30)")
        tint = "rgba(0,176,80,0.08)" if buy else "rgba(224,32,32,0.08)"
        lo, hi = zlo, zhi
        x0 = d.index[-45]
        # เส้นราคาปัจจุบัน (วิ่งตามทิกสด) ให้เห็นชัดว่าห่างโซนแค่ไหน
        fig.add_shape(type="line", x0=x0, x1=x_end, y0=A["price"] + o, y1=A["price"] + o,
                      line=dict(color="#1f6feb", width=1.5))
        fig.add_annotation(x=d.index[0], y=A["price"] + o, xanchor="left", yanchor="bottom", showarrow=False,
                           text=f"ราคา {A['price'] + o:.2f}", font=dict(color="#1f6feb", size=12))
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=min(p["mid"], p["tp1"]) + o, y1=max(p["mid"], p["tp1"]) + o,
                      fillcolor=tint, line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=lo + o, y1=hi + o, fillcolor=fill,
                      line=dict(color=col, width=2.5, dash="solid" if A["status"] == "now" else "dash"))
        fig.add_annotation(x=x_end, y=(lo + hi) / 2 + o, xanchor="right", showarrow=False, font=dict(color=col, size=15),
                           text=f"<b>โซนเข้า {'BUY' if buy else 'SELL'}</b> {lo + o:.2f}-{hi + o:.2f}",
                           bgcolor="rgba(255,255,255,0.92)", bordercolor=col, borderwidth=2)
        for y, nm, c2_ in ((p["sl"], "SL", "#b3372f"), (p["tp1"], "TP1", "#a87a1f"), (p["tp2"], "TP2", "#a87a1f")):
            fig.add_shape(type="line", x0=x0, x1=x_end, y0=y + o, y1=y + o, line=dict(color=c2_, width=1.5, dash="dot"))
            fig.add_annotation(x=x_end, y=y + o, xanchor="right", yanchor="bottom", showarrow=False,
                               text=f"{nm} {y + o:.1f}", font=dict(color=c2_, size=12))
    fig.update_xaxes(range=[d.index[0], x_end])
    # uirevision: ซูม/เลื่อนกราฟแล้วไม่ถูกรีเซ็ตทุกครั้งที่รีเฟรช
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08), uirevision=f"{sym}-{tf}")
    stretch(st.plotly_chart, fig, config=dict(displaylogo=False))
    st.caption(src_note + (f" | 🟢 ทิกสด WebSocket (ล่าสุด {live_age:.0f} วินาทีที่แล้ว)" if live_age is not None
                           else " | 🟡 ยังไม่มีทิกสด ใช้ข้อมูลที่ดึงเป็นรอบ (ตลาดอาจปิด หรือ WebSocket ยังเชื่อมไม่ติด)"))
    st.caption(f"กราฟอัปเดตอัตโนมัติทุก {speed} วินาที | ราคาจาก Yahoo Finance ไม่ใช่ทิกสดของโบรกเกอร์ อาจช้ากว่าและกระโดดเป็นช่วงๆ ใช้ช่องส่วนต่างราคาปรับให้ตรง")


live_view()
