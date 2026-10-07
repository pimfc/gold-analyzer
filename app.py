"""
Gold Analyzer - แอปวิเคราะห์ราคาทอง XAUUSD แบบอัตโนมัติ (ไม่ต้องใช้ MT5)
ดึงราคาล่าสุดจาก Yahoo Finance ทุกไม่กี่วินาที แล้วบอกแนวโน้ม + จุดเข้า Buy/Sell + SL/TP
รันด้วย: streamlit run app.py   (หรือดับเบิลคลิก run.bat)
"""
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Gold Analyzer", page_icon="🪙", layout="centered")

# ค่าตามแต่ละ Timeframe: interval/period ของ Yahoo, ตัวคูณ ATR, ความถี่รีเฟรช (วินาที)
PROFILES = {
    "M1":  dict(interval="1m",  period="5d",   sl=1.5, tp=2.25, refresh=30),
    "M5":  dict(interval="5m",  period="30d",  sl=1.5, tp=2.5,  refresh=45),
    "M15": dict(interval="15m", period="30d",  sl=1.5, tp=3.0,  refresh=60),
    "H1":  dict(interval="1h",  period="180d", sl=1.5, tp=3.0,  refresh=120),
}
SYMBOLS = {"XAUUSD=X (ทองสปอต)": "XAUUSD=X", "GC=F (ทองฟิวเจอร์ส COMEX)": "GC=F"}

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
contract = sb.number_input("Contract size (ออนซ์/lot)", value=100.0, step=1.0)
min_lot = sb.number_input("lot ขั้นต่ำ", value=0.01, step=0.01, format="%.2f")
sb.subheader("แจ้งเตือน Telegram (ไม่บังคับ)")
tg_token = sb.text_input("Bot token", type="password")
tg_chat = sb.text_input("Chat ID")

P = PROFILES[tf]
st_autorefresh(interval=P["refresh"] * 1000, key="auto")


# ---------------- Data & indicators ----------------
@st.cache_data(ttl=15, show_spinner=False)
def load(symbol, interval, period):
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return df


def add_ind(df):
    c = df["close"]
    df["ema_t"] = c.ewm(span=200, adjust=False).mean()
    df["ema_s"] = c.ewm(span=50, adjust=False).mean()
    df["ema_f"] = c.ewm(span=21, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    pc = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
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


def analyse(df):
    d = add_ind(df.copy())
    b1, b6 = d.iloc[-2], d.iloc[-7]          # แท่งที่ปิดแล้ว
    price = float(d.iloc[-1]["close"])
    atr = float(b1["atr"])
    up = b1.close > b1.ema_t and b1.ema_t > b6.ema_t and b1.ema_f > b1.ema_s
    dn = b1.close < b1.ema_t and b1.ema_t < b6.ema_t and b1.ema_f < b1.ema_s
    bias = "up" if up else "down" if dn else "side"

    buy_trig = up and b1.low <= b1.ema_f and b1.close > b1.ema_f and 45 <= b1.rsi <= 68
    sell_trig = dn and b1.high >= b1.ema_f and b1.close < b1.ema_f and 32 <= b1.rsi <= 55

    checks = {
        "ราคาเทียบ EMA200 และ EMA200 ชี้ทิศทางเดียวกัน": up or dn,
        "EMA21 / EMA50 เรียงตามเทรนด์": (b1.ema_f > b1.ema_s) if bias == "up" else (b1.ema_f < b1.ema_s) if bias == "down" else False,
        "ราคาย่อแตะ EMA21 แล้วปิดกลับตามเทรนด์": buy_trig or sell_trig or
            (bias == "up" and b1.low <= b1.ema_f and b1.close > b1.ema_f) or
            (bias == "down" and b1.high >= b1.ema_f and b1.close < b1.ema_f),
        f"RSI อยู่ในช่วงเหมาะสม (ตอนนี้ {b1.rsi:.0f})": (45 <= b1.rsi <= 68) if bias == "up" else (32 <= b1.rsi <= 55) if bias == "down" else False,
    }

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
        sl = mid - s * P["sl"] * atr
        tp1 = mid + s * P["tp"] * atr
        tp2 = mid + s * P["tp"] * 1.6 * atr
        plan = dict(side=side, lo=lo, hi=hi, mid=mid, sl=sl, tp1=tp1, tp2=tp2, sl_dist=abs(mid - sl),
                    rr=abs(tp1 - mid) / abs(mid - sl))
        if side == "buy" and near_res and near_res < tp1:
            warns.append(f"มีแนวต้านที่ {near_res + offset:.2f} ก่อนถึง TP1 ราคาอาจชนแนวนี้ก่อน")
        if side == "sell" and near_sup and near_sup > tp1:
            warns.append(f"มีแนวรับที่ {near_sup + offset:.2f} ก่อนถึง TP1 ราคาอาจชนแนวนี้ก่อน")
        if side == "buy" and b1.rsi > 70:
            warns.append("RSI สูง ระวังไล่ราคา")
        if side == "sell" and b1.rsi < 30:
            warns.append("RSI ต่ำ ระวังไล่ราคา")
        if abs(price - b1.ema_f) > 2.5 * atr and status == "wait":
            warns.append("ราคาอยู่ห่างจากจุดย่อมาก อย่าไล่เข้า รอให้ย้อนกลับมาที่โซน")
    return dict(d=d, bias=bias, status=status, plan=plan, price=price, atr=atr, checks=checks,
                sup=near_sup, res=near_res, warns=warns, bar=d.index[-2])


def lot_for(sl_dist):
    risk_money = balance * risk_pct / 100
    per_lot = sl_dist * contract
    lot = np.floor(risk_money / per_lot / min_lot) * min_lot
    if lot < min_lot:
        actual = min_lot * per_lot
        if actual > risk_money * 2:
            return 0.0, f"lot ขั้นต่ำเสี่ยง ${actual:.2f} เกินกรอบ ${risk_money:.2f} มาก ไม่ควรเข้าไม้นี้ (ลองบัญชี Cent/Micro หรือเพิ่มทุน)"
        return min_lot, f"ใช้ lot ขั้นต่ำ เสี่ยง ${actual:.2f}"
    return round(float(lot), 2), f"เสี่ยงประมาณ ${lot * per_lot:.2f}"


def send_tg(msg):
    if tg_token and tg_chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                          data={"chat_id": tg_chat, "text": msg}, timeout=10)
        except Exception:
            pass


# ---------------- UI ----------------
st.title("🪙 Gold Analyzer")
st.caption("วิเคราะห์ XAUUSD อัตโนมัติจากราคาล่าสุด ไม่ใช่การรับประกันผล ตรวจกราฟจริงและตั้ง SL ทุกไม้")

try:
    df = load(SYMBOLS[sym_label], P["interval"], P["period"])
    if len(df) < 260:
        st.error("ข้อมูลราคาไม่พอ ลองเปลี่ยน Timeframe หรือแหล่งราคา")
        st.stop()
    A = analyse(df)
except Exception as e:
    st.error(f"ดึงราคาไม่สำเร็จ: {e}")
    st.stop()

o = offset
c1, c2, c3 = st.columns(3)
c1.metric("ราคาล่าสุด", f"{A['price'] + o:.2f}")
c2.metric("ATR (ความผันผวน)", f"{A['atr']:.2f}")
c3.metric("อัปเดตล่าสุด", time.strftime("%H:%M:%S"))

label = {"up": "ขาขึ้น 📈", "down": "ขาลง 📉", "side": "ไซด์เวย์ / ไม่ชัด ➖"}[A["bias"]]
st.subheader(f"แนวโน้ม {tf}: {label}")

p = A["plan"]
if A["status"] == "now":
    st.success(f"✅ สัญญาณเข้า {p['side'].upper()} ตอนนี้ (แท่งล่าสุดปิดครบเงื่อนไข)")
elif A["status"] == "wait":
    st.info(f"⏳ แนวโน้มชัด แต่ยังไม่ถึงจุดเข้า รอราคา{'ย่อลง' if p['side']=='buy' else 'เด้งขึ้น'}มาที่โซนด้านล่าง")
else:
    st.warning("⛔ ยังไม่ควรเข้าไม้ แนวโน้มไม่ชัด รอให้เทรนด์ชัดก่อน")

if p:
    lot, lot_note = lot_for(p["sl_dist"])
    t = pd.DataFrame({
        "รายการ": ["ทิศทาง", "โซนเข้า", "Stop loss", "TP1", "TP2", "RR ถึง TP1", "ระยะ SL", "lot ที่แนะนำ"],
        "ค่า": [p["side"].upper(),
                f"{p['lo'] + o:.2f}" if p["lo"] == p["hi"] else f"{p['lo'] + o:.2f} - {p['hi'] + o:.2f}",
                f"{p['sl'] + o:.2f}", f"{p['tp1'] + o:.2f}", f"{p['tp2'] + o:.2f}",
                f"1 : {p['rr']:.1f}", f"{p['sl_dist']:.2f} ดอลลาร์",
                f"{lot:.2f}  ({lot_note})" if lot else "ไม่เปิด: " + lot_note],
    })
    st.dataframe(t, hide_index=True, use_container_width=True)
    for w in A["warns"]:
        st.warning(w)

    if A["status"] == "now" and st.session_state.get("last_alert") != str(A["bar"]):
        st.session_state["last_alert"] = str(A["bar"])
        send_tg(f"สัญญาณ {p['side'].upper()} XAUUSD {tf}\nเข้า ~{p['mid'] + o:.2f}\nSL {p['sl'] + o:.2f}\n"
                f"TP1 {p['tp1'] + o:.2f} | TP2 {p['tp2'] + o:.2f}\nlot {lot:.2f}")

with st.expander("เงื่อนไขที่ใช้ตัดสิน", expanded=False):
    for k, v in A["checks"].items():
        st.write(("✅ " if v else "❌ ") + k)
    st.write(f"แนวรับใกล้สุด: {A['sup'] + o:.2f}" if A["sup"] else "แนวรับใกล้สุด: -")
    st.write(f"แนวต้านใกล้สุด: {A['res'] + o:.2f}" if A["res"] else "แนวต้านใกล้สุด: -")

# กราฟ
d = A["d"].tail(120)
fig = go.Figure(go.Candlestick(x=d.index, open=d.open + o, high=d.high + o, low=d.low + o, close=d.close + o, name="ราคา"))
for col, nm in (("ema_t", "EMA200"), ("ema_s", "EMA50"), ("ema_f", "EMA21")):
    fig.add_trace(go.Scatter(x=d.index, y=d[col] + o, name=nm, line=dict(width=1.2)))
if p:
    for y, nm, colr in ((p["lo"], "เข้า", "#1b7a62"), (p["sl"], "SL", "#b3372f"), (p["tp1"], "TP1", "#a87a1f")):
        fig.add_hline(y=y + o, line_dash="dot", line_color=colr, annotation_text=nm)
fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"หน้านี้รีเฟรชอัตโนมัติทุก {P['refresh']} วินาที | ราคาจาก Yahoo Finance อาจดีเลย์และไม่เท่าโบรกเกอร์ ใช้ช่องส่วนต่างราคาปรับให้ตรง")
