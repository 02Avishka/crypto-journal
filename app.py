import os
import requests
import pandas as pd
import streamlit as st

# =========================================================
# Page config MUST be first Streamlit call
# =========================================================
st.set_page_config(page_title="Crypto Trading Journal (Local)", layout="wide")

# =========================================================
# Light Soft UI + Visible Inputs (Trader Friendly)
# =========================================================
st.markdown("""
<style>
.stApp {
  background: radial-gradient(1200px circle at 10% 10%, rgba(14,165,164,0.10), transparent 45%),
              radial-gradient(900px circle at 90% 15%, rgba(59,130,246,0.08), transparent 40%),
              radial-gradient(900px circle at 40% 90%, rgba(34,197,94,0.06), transparent 40%),
              #F7FAFC;
}
.block-container { padding-top: 2.0rem; }

div[data-testid="stForm"],
div[data-testid="stDataFrame"],
div[data-testid="stMetric"],
div[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] > div {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(15,23,42,0.08);
  backdrop-filter: blur(8px);
}
div[data-testid="stForm"] { padding: 18px 18px 10px 18px !important; }

hr { border: none; height: 1px; background: rgba(15,23,42,0.10); }

.stButton > button {
  border-radius: 14px !important;
  border: 1px solid rgba(14,165,164,0.35) !important;
  background: rgba(14,165,164,0.10) !important;
  box-shadow: 0 8px 18px rgba(15,23,42,0.08) !important;
}
.stButton > button:hover { background: rgba(14,165,164,0.16) !important; }

label, .stMarkdown, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
  color: rgba(15,23,42,0.85) !important;
  font-weight: 600 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stTextArea"] textarea {
  background: rgba(255,255,255,0.98) !important;
  border: 1.5px solid rgba(15,23,42,0.22) !important;
  box-shadow: inset 0 1px 0 rgba(15,23,42,0.06) !important;
  border-radius: 14px !important;
  padding: 10px 14px !important;
  font-size: 15px !important;
}

div[data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,0.98) !important;
  border: 1.5px solid rgba(15,23,42,0.22) !important;
  border-radius: 14px !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stDateInput"] input:focus {
  outline: none !important;
  border: 2px solid rgba(14,165,164,0.70) !important;
  box-shadow: 0 0 0 4px rgba(14,165,164,0.18) !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
  color: rgba(15,23,42,0.45) !important;
}

div[data-testid="stAlert"] { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("📒 Crypto Trading Journal (Local)")

# =========================================================
# Settings
# =========================================================
LOCAL_CSV = "trades.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# =========================================================
# Helpers
# =========================================================
def ask_ollama(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")

def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False),
        errors="coerce",
    )

def normalize_sheet_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw.columns = [c.strip() for c in df_raw.columns]

    col_map = {
        "Date": "date",
        "Trading Pairs": "pair",
        "Entry Price": "entry",
        "Exit Price": "exit",
        "Position": "position",
        "Profit / Loss": "profit_loss",
        "Reason [Emotion]": "emotion",
        "Lesson": "lesson",
    }

    missing = [c for c in col_map.keys() if c not in df_raw.columns]
    if missing:
        raise ValueError(f"CSV columns mismatch. Missing: {missing}")

    df = df_raw.rename(columns=col_map).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    df["entry"] = to_number(df["entry"]).fillna(0)
    df["exit"] = to_number(df["exit"]).fillna(0)
    df["profit_loss"] = to_number(df["profit_loss"]).fillna(0)

    for c in ["pair", "position", "emotion", "lesson"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True)

def load_local_csv() -> pd.DataFrame:
    if not os.path.exists(LOCAL_CSV):
        return pd.DataFrame(columns=["date","pair","entry","exit","position","profit_loss","emotion","lesson"])

    df = pd.read_csv(LOCAL_CSV)

    for col in ["date","pair","entry","exit","position","profit_loss","emotion","lesson"]:
        if col not in df.columns:
            df[col] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for col in ["pair","position","emotion","lesson"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ["entry","exit","profit_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.reset_index(drop=True)

def save_local_csv(df: pd.DataFrame):
    df_to_save = df.copy()
    df_to_save["date"] = pd.to_datetime(df_to_save["date"]).dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(LOCAL_CSV, index=False)

# =========================================================
# 0) Add Trade (Trader UI Grid)
# =========================================================
st.subheader("0) Add a Trade (Trader Entry Panel)")

with st.form("add_trade_form", clear_on_submit=True):
    r1c1, r1c2, r1c3, r1c4 = st.columns([1.1, 1.6, 1.1, 1.2])
    with r1c1:
        t_date = st.date_input("Date")
    with r1c2:
        t_pair = st.text_input("Trading Pair (e.g., BTCUSDT)")
    with r1c3:
        t_position = st.selectbox("Position", ["Long", "Short"])
    with r1c4:
        t_emotion = st.text_input("Reason / Emotion (e.g., Calm, FOMO)")

    r2c1, r2c2, r2c3, r2c4 = st.columns([1.2, 1.2, 1.2, 1.6])
    with r2c1:
        t_entry = st.number_input("Entry", min_value=0.0, value=0.0, step=0.5)
    with r2c2:
        t_exit = st.number_input("Exit", min_value=0.0, value=0.0, step=0.5)
    with r2c3:
        t_pl = st.number_input("Profit / Loss", value=0.0, step=1.0)
    with r2c4:
        t_lesson = st.text_input("Lesson / Note")

    a1, a2, a3 = st.columns([1.3, 1.2, 3.5])
    with a1:
        add_submitted = st.form_submit_button("➕ Add Trade")
    with a2:
        quick_clear = st.form_submit_button("🧹 Clear (Reload)")
    with a3:
        st.caption("Tip: Pair uppercase (BTCUSDT). Emotion short keywords (Calm/FOMO/Fear).")

if quick_clear:
    st.rerun()

if add_submitted:
    df_local = load_local_csv()
    new_row = pd.DataFrame([{
        "date": pd.to_datetime(str(t_date)),
        "pair": (t_pair or "").strip(),
        "entry": float(t_entry),
        "exit": float(t_exit),
        "position": (t_position or "").strip(),
        "profit_loss": float(t_pl),
        "emotion": (t_emotion or "").strip(),
        "lesson": (t_lesson or "").strip(),
    }])

    df_local = pd.concat([df_local, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    save_local_csv(df_local)
    st.success(f"Saved ✅ Trade added to {LOCAL_CSV}")
    st.rerun()

st.divider()

# =========================================================
# 1) Upload CSV OR Use Local
# =========================================================
st.subheader("1) Upload your trades.csv (or use local saved trades.csv)")
uploaded = st.file_uploader("Upload trades.csv", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        df = normalize_sheet_df(df_raw)
        source = "Uploaded CSV (Google Sheets)"
    except Exception as e:
        st.error(f"CSV read/format error: {e}")
        st.stop()
else:
    df = load_local_csv()
    source = "Local trades.csv"

st.info(f"Data source: {source}")

# =========================================================
# Main dashboard (only if data exists)
# =========================================================
if df is None or len(df) == 0:
    st.warning("No trades yet. Use 'Add a Trade' form above OR upload a CSV with rows.")
    st.caption("Tip: Google Sheets -> File -> Download -> CSV (.csv) -> Upload කරන්න.")
    st.stop()

# =========================================================
# Edit / Delete (LOCAL)
# =========================================================
st.subheader("✏️ Edit / 🗑️ Delete a Trade (Local trades.csv)")
df_local = load_local_csv()

if len(df_local) == 0:
    st.info("No local trades to edit/delete.")
else:
    labels = [f"#{i} | {r['date'].date()} | {r['pair']} | {r['position']} | P/L {r['profit_loss']}"
              for i, r in df_local.iterrows()]

    selected_label = st.selectbox("Select a local trade", labels)
    selected_index = int(selected_label.split("|")[0].replace("#", "").strip())
    row = df_local.loc[selected_index]

    tab1, tab2 = st.tabs(["✏️ Edit", "🗑️ Delete"])

    with tab1:
        with st.form("edit_trade_form"):
            e1, e2, e3, e4 = st.columns(4)

            with e1:
                edit_date = st.date_input("Date", value=row["date"].date(), key="ed_date")
                edit_pair = st.text_input("Trading Pair", value=str(row["pair"]), key="ed_pair")

            with e2:
                edit_entry = st.number_input("Entry", min_value=0.0, value=float(row["entry"]), step=0.5, key="ed_entry")
                edit_exit = st.number_input("Exit", min_value=0.0, value=float(row["exit"]), step=0.5, key="ed_exit")

            with e3:
                pos_default = "Long" if str(row["position"]).strip().lower() != "short" else "Short"
                edit_position = st.selectbox("Position", ["Long", "Short"], index=0 if pos_default == "Long" else 1, key="ed_pos")
                edit_pl = st.number_input("Profit / Loss", value=float(row["profit_loss"]), step=1.0, key="ed_pl")

            with e4:
                edit_emotion = st.text_input("Emotion", value=str(row["emotion"]), key="ed_emo")
                edit_lesson = st.text_input("Lesson", value=str(row["lesson"]), key="ed_lesson")

            save_edit = st.form_submit_button("💾 Save changes")

        if save_edit:
            df_local.at[selected_index, "date"] = pd.to_datetime(str(edit_date))
            df_local.at[selected_index, "pair"] = (edit_pair or "").strip()
            df_local.at[selected_index, "entry"] = float(edit_entry)
            df_local.at[selected_index, "exit"] = float(edit_exit)
            df_local.at[selected_index, "position"] = (edit_position or "").strip()
            df_local.at[selected_index, "profit_loss"] = float(edit_pl)
            df_local.at[selected_index, "emotion"] = (edit_emotion or "").strip()
            df_local.at[selected_index, "lesson"] = (edit_lesson or "").strip()

            df_local = df_local.sort_values("date").reset_index(drop=True)
            save_local_csv(df_local)
            st.success("Updated ✅ Trade saved.")
            st.rerun()

    with tab2:
        confirm = st.checkbox("I confirm delete", value=False, key="confirm_delete")
        if st.button("❌ Delete selected trade", disabled=not confirm):
            df_after = df_local.drop(index=selected_index).reset_index(drop=True)
            save_local_csv(df_after)
            st.success("Deleted ✅")
            st.rerun()

st.divider()

# =========================================================
# Trades Table + Quick Stats
# =========================================================
st.write("### 📄 Trades Table")
st.dataframe(df, use_container_width=True)

st.write("### 📌 Quick Stats")
total_pnl = float(df["profit_loss"].sum())
win_rate = float((df["profit_loss"] > 0).mean() * 100)

k1, k2, k3 = st.columns(3)
k1.metric("Total P/L", f"{total_pnl:.2f}")
k2.metric("Win Rate", f"{win_rate:.1f}%")
k3.metric("Total Trades", f"{len(df)}")

st.divider()

# =========================================================
# 2) Charts (Daily) + Equity Curve
# =========================================================
st.subheader("2) Charts (Daily)")

df2 = df.copy()
df2["day"] = pd.to_datetime(df2["date"]).dt.date

starting_balance = st.number_input(
    "Starting Balance ($)",
    min_value=0.0,
    value=100.0,
    step=50.0
)

daily_pl = df2.groupby("day")["profit_loss"].sum()
daily_cum_pl = daily_pl.cumsum()
equity_curve = starting_balance + daily_cum_pl

st.write("✅ Trading Journey (Lifetime) — Daily Cumulative P/L")
st.line_chart(daily_cum_pl)

st.write("✅ Equity Curve (Starting Balance + Cumulative P/L)")
st.line_chart(equity_curve)

st.write("✅ Daily Profit / Loss")
st.bar_chart(daily_pl)

st.divider()

# =========================================================
# Risk Management Calculator (NO Qty display)
# =========================================================
st.subheader("🧮 Risk Management Calculator (Futures + Leverage)")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=1000.0, step=10.0, key="rm_bal")
with c2:
    risk_percent = st.number_input("Risk %", min_value=0.0, value=1.0, step=0.1, key="rm_risk")
with c3:
    entry = st.number_input("Entry Price", min_value=0.0, value=65000.0, step=10.0, key="rm_entry")
with c4:
    stop_loss = st.number_input("Stop Loss Price", min_value=0.0, value=64000.0, step=10.0, key="rm_sl")
with c5:
    leverage = st.number_input("Leverage (x)", min_value=1.0, value=10.0, step=1.0, key="rm_lev")
with c6:
    contract_size = st.number_input("Contract Size (optional)", min_value=0.000001, value=1.0, step=0.5, key="rm_cs")

f1, f2, f3, f4 = st.columns(4)
with f1:
    est_fee_percent = st.number_input("Fees % (optional)", min_value=0.0, value=0.0, step=0.01, key="rm_fee")
with f2:
    slippage_percent = st.number_input("Slippage % (optional)", min_value=0.0, value=0.0, step=0.01, key="rm_slip")
with f3:
    _ = st.selectbox("Mode", ["Isolated", "Cross"], key="rm_mode")
with f4:
    _ = st.selectbox("Contract Type", ["Linear", "Inverse"], key="rm_ct")

risk_amount = balance * (risk_percent / 100.0)
sl_distance = abs(entry - stop_loss)

if entry <= 0 or stop_loss <= 0:
    st.error("Entry සහ Stop Loss values > 0 විය යුතුයි.")
elif sl_distance == 0:
    st.error("Stop Loss distance = 0. Entry සහ Stop Loss වෙනස් විය යුතුයි.")
else:
    qty_coin = risk_amount / sl_distance  # internal only
    _qty_contracts = qty_coin / contract_size

    notional = qty_coin * entry
    required_margin = notional / leverage
    effective_leverage = notional / balance if balance > 0 else 0
    sl_percent = (sl_distance / entry) * 100

    fee_cost = notional * (est_fee_percent / 100.0) * 2
    slip_cost = notional * (slippage_percent / 100.0)
    est_total_cost = fee_cost + slip_cost

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Amount ($)", f"{risk_amount:.2f}")
    m2.metric("SL Distance ($)", f"{sl_distance:.2f}")
    m3.metric("SL Distance (%)", f"{sl_percent:.2f}%")
    m4.metric("Position Notional ($)", f"{notional:.2f}")

    x1, x2, x3 = st.columns(3)
    x1.metric("Leverage (x)", f"{leverage:.1f}x")
    x2.metric("Required Margin ($)", f"{required_margin:.2f}")
    x3.metric("Effective Leverage (x)", f"{effective_leverage:.2f}x")

    if required_margin > balance and balance > 0:
        st.warning("⚠️ Required margin > Account balance. Leverage වැඩි කරන්න හෝ risk % අඩු කරන්න හෝ SL distance වෙනස් කරන්න.")

    if est_fee_percent > 0 or slippage_percent > 0:
        st.caption(f"Estimated fees + slippage cost (rough): ${est_total_cost:.2f}")

st.divider()

# =========================================================
# AI Analysis (Ollama)
# =========================================================
st.subheader("🤖 AI Analysis (Weekly / Monthly)")

report_type = st.selectbox("Select report type", ["Weekly", "Monthly"], key="rep_type")
start = st.date_input("Start date", value=pd.to_datetime(df["date"]).min().date(), key="start_date")
end = st.date_input("End date", value=pd.to_datetime(df["date"]).max().date(), key="end_date")

if st.button("Generate AI Report", key="gen_report"):
    mask = (pd.to_datetime(df["date"]).dt.date >= start) & (pd.to_datetime(df["date"]).dt.date <= end)
    dff = df.loc[mask].copy()

    if len(dff) == 0:
        st.error("No trades found in this date range.")
    else:
        rows = []
        for _, r in dff.iterrows():
            rows.append(
                f"{pd.to_datetime(r['date']).date()} | {r['pair']} | {r['position']} | "
                f"Entry: {r['entry']} | Exit: {r['exit']} | "
                f"P/L: {r['profit_loss']} | Emotion: {r['emotion']} | Lesson: {r['lesson']}"
            )
        trades_text = "\n".join(rows[:300])

        prompt_lines = [
            "You are a crypto trading journal coach.",
            f"Generate a {report_type.lower()} report using the trades below.",
            "",
            "Rules:",
            "- Keep it structured with headings.",
            "- Use simple Sinhala + some English trading terms.",
            "- Do NOT give financial advice. Focus on journaling, process, discipline, and risk control.",
            "- Be honest, direct, and practical.",
            "",
            "MUST INCLUDE these sections:",
            "",
            "1) Performance Summary",
            "- Total P/L, win rate feeling (based on trades), best day/worst day (if possible)",
            "",
            "2) Discipline Score (0–10)",
            "- Give a score and explain WHY (3 short bullet reasons)",
            "- Base it on: consistent risk, following plan, emotional control, avoiding overtrading",
            "",
            "3) Top 3 Repeated Mistakes",
            "- Identify the 3 most repeated mistakes from the journal fields and outcomes",
            "- For each mistake: (a) Evidence pattern (b) Why it happens (c) Fix rule (one sentence)",
            "",
            "4) Emotion Patterns",
            "- Which emotions are linked to losses/wins?",
            "- 2 rules to manage the worst emotion",
            "",
            "5) Risk Management Feedback",
            "- Comment on risk consistency and stop-loss discipline",
            "- Suggest 2 concrete improvements (process-based, not money advice)",
            "",
            f"6) Action Plan (Next {report_type})",
            "- 3–5 action items (very specific)",
            "- Include a pre-trade checklist of 5 items",
            "",
            "Trades:",
            trades_text
        ]

        prompt = "\n".join(prompt_lines)

        try:
            with st.spinner("Thinking with Ollama..."):
                report = ask_ollama(prompt)
            st.success("Report generated ✅")
            st.write(report)
        except requests.exceptions.ConnectionError:
            st.error("❌ Ollama connect වෙන්නේ නැහැ. `ollama serve` run කරලා බලන්න.")
        except Exception as e:
            st.error(f"❌ Ollama error: {e}")