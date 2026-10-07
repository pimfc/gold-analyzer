"""
Gold Analyzer - แอปวิเคราะห์ราคาทอง XAUUSD แบบอัตโนมัติ (ไม่ต้องใช้ MT5)
ดึงราคาล่าสุดจาก Yahoo Finance ทุกไม่กี่วินาที แล้วบอกแนวโน้ม + จุดเข้า Buy/Sell + SL/TP
รันด้วย: streamlit run app.py   (หรือดับเบิลคลิก run.bat)
"""
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

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
sb.subheader("แหล่งราคาสำรอง (ไม่บังคับ)")
try:
    _sec = st.secrets.get("TWELVE_KEY", "")
except Exception:
    _sec = ""
td_key = sb.text_input("Twelve Data API key", value=_sec, type="password",
                       help="สมัครฟรีที่ twelvedata.com ใช้เมื่อ Yahoo ไม่ส่งข้อมูล (เซิร์ฟเวอร์คลาวด์มักถูก Yahoo จำกัด)")

speed = sb.selectbox("ความเร็วอัปเดตกราฟ (วินาที)", [5, 10, 30, 60], index=1,
                     help="ยิ่งถี่ยิ่งเสี่ยงโดน Yahoo จำกัดการเข้าถึง ถ้าข้อมูลหาย ให้เพิ่มเป็น 10-30 วินาที")
P = PROFILES[tf]


# ---------------- Data & indicators ----------------
TD_INTERVAL = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}


def _clean(df):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].astype(float).dropna()
    return df[~df.index.duplicated()].sort_index()


@st.cache_data(ttl=4, show_spinner=False)
def load(symbol, interval, period, td_key):
    """คืน (df, ข้อความอธิบาย) ลอง Yahoo ทั้งสองสัญลักษณ์ก่อน แล้วค่อย Twelve Data"""
    notes = []
    for sym in dict.fromkeys([symbol, "GC=F", "XAUUSD=X"]):
        try:
            df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and len(df) >= 260:
                return _clean(df), (f"ใช้ข้อมูล Yahoo ({sym})" + ("" if sym == symbol else " แทนแหล่งที่เลือก"))
            notes.append(f"Yahoo {sym}: ได้ {0 if df is None else len(df)} แท่ง")
        except Exception as e:
            notes.append(f"Yahoo {sym}: {type(e).__name__}")
    if td_key:
        try:
            r = requests.get("https://api.twelvedata.com/time_series", timeout=15, params=dict(
                symbol="XAU/USD", interval=TD_INTERVAL[interval], outputsize=600, timezone="UTC", apikey=td_key)).json()
            if "values" in r:
                df = pd.DataFrame(r["values"])
                df.index = pd.to_datetime(df.pop("datetime"))
                df = _clean(df)
                if len(df) >= 260:
                    return df, "ใช้ข้อมูล Twelve Data (XAU/USD)"
                notes.append(f"Twelve Data: ได้ {len(df)} แท่ง")
            else:
                notes.append(f"Twelve Data: {r.get('message', 'ไม่มีข้อมูล')}")
        except Exception as e:
            notes.append(f"Twelve Data: {type(e).__name__}")
    return None, " | ".join(notes)


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
    # สัญญาณย้อนหลังทุกแท่ง (ใช้เงื่อนไขเดียวกับสัญญาณปัจจุบัน) ไว้วาดลูกศรบนกราฟ
    up_t = (df.close > df.ema_t) & (df.ema_t > df.ema_t.shift(5)) & (df.ema_f > df.ema_s)
    dn_t = (df.close < df.ema_t) & (df.ema_t < df.ema_t.shift(5)) & (df.ema_f < df.ema_s)
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

@st.fragment(run_every=speed)
def live_view():
    df, src_note = load(SYMBOLS[sym_label], P["interval"], P["period"], td_key)
    if df is None:
        st.error("ดึงข้อมูลราคาไม่ได้ในตอนนี้")
        st.code(src_note or "ไม่มีรายละเอียด")
        st.info("ถ้าเห็นว่า Yahoo ได้ 0 แท่งทุกตัว แปลว่า Yahoo ปิดกั้นเซิร์ฟเวอร์คลาวด์ ให้ใส่ Twelve Data API key (ฟรี) "
                "ที่แถบข้างซ้ายหัวข้อ 'แหล่งราคาสำรอง' หรือรอสักครู่แล้วลองใหม่")
        return
    try:
        A = analyse(df)
    except Exception as e:
        st.error(f"วิเคราะห์ข้อมูลไม่สำเร็จ: {e}")
        return

    o = offset
    c1, c2, c3 = st.columns(3)
    prev = st.session_state.get("prev_price")
    c1.metric("ราคาล่าสุด", f"{A['price'] + o:.2f}", None if prev is None else f"{A['price'] - prev:+.2f}")
    st.session_state["prev_price"] = A["price"]
    c2.metric("ATR (ความผันผวน)", f"{A['atr']:.2f}")
    c3.metric("อัปเดตล่าสุด", datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%H:%M:%S"))

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
    step = d.index[-1] - d.index[-2]
    x_end = d.index[-1] + step * 14
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
        lo, hi = p["lo"], p["hi"]
        if hi - lo < 0.3 * A["atr"]:                       # โซนบางเกินไปให้ขยายให้มองเห็น
            m = (lo + hi) / 2
            lo, hi = m - 0.15 * A["atr"], m + 0.15 * A["atr"]
        x0 = d.index[-45]
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=min(p["mid"], p["tp1"]) + o, y1=max(p["mid"], p["tp1"]) + o,
                      fillcolor=tint, line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=lo + o, y1=hi + o, fillcolor=fill,
                      line=dict(color=col, width=2.5, dash="solid" if A["status"] == "now" else "dash"))
        fig.add_annotation(x=x_end, y=(lo + hi) / 2 + o, xanchor="right", showarrow=False, font=dict(color=col, size=13),
                           text=f"<b>โซน {'BUY' if buy else 'SELL'}</b> {lo + o:.1f}-{hi + o:.1f}", bgcolor="rgba(255,255,255,0.85)")
        for y, nm, c2_ in ((p["sl"], "SL", "#b3372f"), (p["tp1"], "TP1", "#a87a1f"), (p["tp2"], "TP2", "#a87a1f")):
            fig.add_shape(type="line", x0=x0, x1=x_end, y0=y + o, y1=y + o, line=dict(color=c2_, width=1.5, dash="dot"))
            fig.add_annotation(x=x_end, y=y + o, xanchor="right", yanchor="bottom", showarrow=False,
                               text=f"{nm} {y + o:.1f}", font=dict(color=c2_, size=12))
    fig.update_xaxes(range=[d.index[0], x_end])
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(src_note)
    st.caption(f"กราฟอัปเดตอัตโนมัติทุก {speed} วินาที | ราคาจาก Yahoo Finance ไม่ใช่ทิกสดของโบรกเกอร์ อาจช้ากว่าและกระโดดเป็นช่วงๆ ใช้ช่องส่วนต่างราคาปรับให้ตรง")


live_view()
