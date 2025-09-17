import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AR Agentic AI – Trading & Distribution", layout="wide", page_icon="🥩")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    .big-number {font-size: 36px; font-weight: 700; margin-bottom: -10px;}
    .small-label {font-size: 12px; color: #666;}
    .kpi-card {padding: 16px; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); background: white;}
    .section {padding: 8px 12px; border-left: 4px solid #16a34a; background: #f6fff8; border-radius: 8px; margin: 12px 0;}
    .pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; margin-right:6px;}
    </style>
    """, unsafe_allow_html=True
)

# ---------- Data Load ----------
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)

scenarios = load_csv("scenarios.csv")
meating_points = load_csv("meating_points.csv")
trading_orders = load_csv("trading_orders.csv")
processing_mix = load_csv("processing_mix.csv")
fridge_econ = load_csv("fridge_economics.csv").iloc[0].to_dict()

# Sidebar assumptions
st.sidebar.header("Global Assumptions")
procurement_per_kg = st.sidebar.number_input("Procurement per kg (AUS origin, SGD)", 0.0, 100.0, 6.50, 0.10)
ocean_freight_per_kg = st.sidebar.number_input("Ocean freight per kg (AU→SG)", 0.0, 50.0, 0.80, 0.05)
customs_default = float(scenarios.loc[scenarios['scenario']=="Baseline","customs_per_kg"].iloc[0])
customs_per_kg = st.sidebar.number_input("Customs & duties per kg", 0.0, 20.0, customs_default, 0.05)

scenario = st.sidebar.selectbox("Scenario", scenarios["scenario"].tolist(), index=0)
scenario_row = scenarios[scenarios["scenario"] == scenario].iloc[0]

st.sidebar.caption("Fridge Model")
st.sidebar.code(f"Model: {fridge_econ['model']} | CAPEX {fridge_econ['capex_sgd']} SGD | Depreciation {fridge_econ['depreciation_months']}m")

st.title("Aliyah Rizq – Agentic AI Demo")
st.subheader("Simulating an LLM/LAM planning loop for a trading and distribution workflow")

tab1, tab2, tab3, tab4 = st.tabs(["Trading Biz (Core)", "Distribution Biz (C1.1)", "Processing Biz (C1.2)", "Agentic Planner"])

# ---------- Helpers ----------
def kpi(label, value, suffix=""):
    with st.container():
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-number">{value:,.2f}{suffix}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-label">{label}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Trading Biz ----------
with tab1:
    st.markdown("### Purchase Orders & Landed Cost")

    # Apply scenario freight multiplier/customs
    tr = trading_orders.copy()
    tr["freight_per_kg"] = tr["freight_per_kg"] * float(scenario_row["freight_multiplier"])
    tr["landed_cost_per_kg"] = tr["unit_price_sgd"] + tr["freight_per_kg"] + float(scenario_row["customs_per_kg"])

    total_qty = tr["qty_kg"].sum()
    avg_landed = (tr["landed_cost_per_kg"] * tr["qty_kg"]).sum() / max(total_qty, 1)
    eta_days = pd.to_datetime(tr["delivery_eta"]) - pd.to_datetime(tr["order_date"])
    avg_eta = eta_days.dt.days.mean()
    open_pos = (tr["status"] != "Delivered").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total Ordered (kg)", total_qty)
    with c2: kpi("Avg Landed (SGD/kg)", avg_landed)
    with c3: kpi("Avg ETA (days)", avg_eta)
    with c4: kpi("Open POs", open_pos)

    st.dataframe(tr, use_container_width=True, height=360)

    vol = tr.groupby("sku", as_index=False)["qty_kg"].sum()
    fig = px.bar(vol, x="sku", y="qty_kg", title="Volume by SKU (kg)")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Distribution Biz (C1.1) ----------
with tab2:
    st.markdown("### Channel Assumptions")
    demand_bump = float(scenario_row["demand_shock_pct"])
    price_adj = float(scenario_row["price_adj_pct"])
    spoilage_pct = float(scenario_row["spoilage_rate_pct"])
    cycle_days = int(scenario_row["procurement_cycle_days"])

    colA, colB, colC = st.columns(3)
    with colA:
        home_vol = st.number_input("Monthly volume – Home (kg)", 0, 100000, 1200, 50)
        home_price = st.number_input("Price/kg – Home", 0.0, 100.0, 14.0, 0.1) * (1+price_adj)
        home_dist = st.number_input("Distribution cost/kg – Home", 0.0, 20.0, 1.50, 0.05)
    with colB:
        mini_vol = st.number_input("Monthly volume – Mini (kg)", 0, 100000, 1800, 50)
        mini_price = st.number_input("Price/kg – Mini", 0.0, 100.0, 13.0, 0.1) * (1+price_adj)
        mini_dist = st.number_input("Distribution cost/kg – Mini", 0.0, 20.0, 1.20, 0.05)
    with colC:
        rest_vol = st.number_input("Monthly volume – Restaurant (kg)", 0, 200000, 2500, 50)
        rest_price = st.number_input("Price/kg – Restaurant", 0.0, 100.0, 12.5, 0.1) * (1+price_adj)
        rest_dist = st.number_input("Distribution cost/kg – Restaurant", 0.0, 20.0, 1.00, 0.05)

    landed = procurement_per_kg + ocean_freight_per_kg * float(scenario_row["freight_multiplier"]) + customs_per_kg
    ch = pd.DataFrame([
        {"channel":"Home","monthly_volume": home_vol*(1+demand_bump),"price": home_price,"dist_cost": home_dist},
        {"channel":"Mini","monthly_volume": mini_vol*(1+demand_bump),"price": mini_price,"dist_cost": mini_dist},
        {"channel":"Restaurant","monthly_volume": rest_vol*(1+demand_bump),"price": rest_price,"dist_cost": rest_dist},
    ])

    ch["revenue"] = ch["monthly_volume"] * ch["price"]
    ch["product_cogs"] = ch["monthly_volume"] * landed * (1 + spoilage_pct/100)
    ch["distribution_costs"] = ch["monthly_volume"] * ch["dist_cost"]
    ch["gross_profit"] = ch["revenue"] - ch["product_cogs"] - ch["distribution_costs"]
    gp_margin = ch["gross_profit"].sum() / max(ch["revenue"].sum(), 1)

    st.markdown(f"**Landed cost per kg:** SGD {landed:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp; **Spoilage:** {spoilage_pct:.1f}%  &nbsp;&nbsp;|&nbsp;&nbsp; **Procurement cycle:** {cycle_days} days")

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Revenue (monthly, SGD)", ch["revenue"].sum())
    with k2: kpi("Product COGS (SGD)", ch["product_cogs"].sum())
    with k3: kpi("Distribution Costs (SGD)", ch["distribution_costs"].sum())
    with k4: kpi("Gross Profit (SGD)", ch["gross_profit"].sum())

    st.dataframe(ch, use_container_width=True, height=300)

    st.markdown("#### Working Capital (Inventory only)")
    avg_daily_cogs = (ch["product_cogs"].sum() / max(30,1))
    inventory_value = avg_daily_cogs * (cycle_days / 2)  # simple cycle-stock model
    kpi("Inventory Value (avg, SGD)", inventory_value)

    # Fridge ROI for Meating Points
    st.markdown("### Fridge Deployment ROI (Meating Points)")
    mp = meating_points.copy()
    mp["base_margin_per_kg"] = (mp["price_per_kg"] - (landed + mp["dist_cost_per_kg"])).clip(lower=0.0)

    uplift = float(fridge_econ["uplift_demand_pct"]) / 100.0
    spoilage_reduction = float(fridge_econ["spoilage_reduction_pct"]) / 100.0
    base_spoilage = spoilage_pct/100.0

    monthly_base_sales = mp["daily_demand_kg"] * 30
    monthly_uplift_sales = monthly_base_sales * uplift

    mp["monthly_margin_gain"] = (monthly_uplift_sales * mp["base_margin_per_kg"]) + \
                                 (monthly_base_sales * mp["base_margin_per_kg"] * base_spoilage * spoilage_reduction)

    monthly_costs = (float(fridge_econ["capex_sgd"]) / float(fridge_econ["depreciation_months"])) + \
                    float(fridge_econ["maintenance_sgd_month"]) + float(fridge_econ["energy_sgd_month"]) - \
                    float(fridge_econ["rental_income_sgd_month"])

    mp["net_monthly_gain"] = mp["monthly_margin_gain"] - monthly_costs
    mp["payback_months"] = np.where(mp["net_monthly_gain"]>0, float(fridge_econ["capex_sgd"]) / mp["net_monthly_gain"], np.nan)
    mp_sorted = mp.sort_values("payback_months")

    st.dataframe(mp_sorted[["mp_id","area","channel","daily_demand_kg","base_margin_per_kg","net_monthly_gain","payback_months"]].head(8), use_container_width=True)
    fig2 = px.bar(mp_sorted.head(10), x="mp_id", y="payback_months", color="channel", title="Top Meating Points by Payback (months)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Processing Biz (C1.2) ----------
with tab3:
    st.markdown("### Processing Mix & Fees")
    df = processing_mix.copy()
    row = df.iloc[-1]
    input_vol = st.number_input("Monthly input volume (kg)", 0, 500000, int(row['input_volume_kg']), 500)
    fee_cut = st.number_input("Contract fee/kg – Cutting", 0.0, 50.0, float(row["fee_cutting_per_kg"]), 0.1)
    fee_pre = st.number_input("Contract fee/kg – Preprocessed", 0.0, 50.0, float(row["fee_preprocessed_per_kg"]), 0.1)
    fee_rte = st.number_input("Contract fee/kg – Ready-to-Eat", 0.0, 50.0, float(row["fee_rte_per_kg"]), 0.1)

    pct_cut = st.slider("% Cutting", 0, 100, int(row["pct_cutting"]*100))
    pct_pre = st.slider("% Preprocessed", 0, 100, int(row["pct_preprocessed"]*100))
    pct_rte = max(0, 100 - pct_cut - pct_pre)
    st.caption(f"Computed % Ready-to-Eat: **{pct_rte}%**")

    y_cut = st.slider("Yield – Cutting (%)", 80, 100, int(row["yield_cutting_pct"]))
    y_pre = st.slider("Yield – Preprocessed (%)", 80, 100, int(row["yield_preprocessed_pct"]))
    y_rte = st.slider("Yield – Ready-to-Eat (%)", 70, 100, int(row["yield_rte_pct"]))

    mix = pd.DataFrame([
        {"product":"Cutting","in_kg": input_vol*(pct_cut/100),"yield_pct": y_cut,"out_kg": input_vol*(pct_cut/100)*(y_cut/100),"fee_per_kg": fee_cut},
        {"product":"Preprocessed","in_kg": input_vol*(pct_pre/100),"yield_pct": y_pre,"out_kg": input_vol*(pct_pre/100)*(y_pre/100),"fee_per_kg": fee_pre},
        {"product":"Ready-to-Eat","in_kg": input_vol*(pct_rte/100),"yield_pct": y_rte,"out_kg": input_vol*(pct_rte/100)*(y_rte/100),"fee_per_kg": fee_rte},
    ])
    mix["processing_spend"] = mix["in_kg"] * mix["fee_per_kg"]
    st.dataframe(mix, use_container_width=True)

    total_processing_spend = mix["processing_spend"].sum()
    kpi("Total processing spend / month (SGD)", total_processing_spend)

    fig3 = px.pie(mix, names="product", values="processing_spend", title="Processing Spend Mix")
    st.plotly_chart(fig3, use_container_width=True)

# ---------- Agentic Planner ----------
with tab4:
    st.markdown("### LLM/LAM Planning Loop (Simulated)")
    target_margin = st.slider("Target GP Margin", 0.05, 0.6, 0.30, 0.01)
    max_payback_months = st.slider("Max acceptable payback (fridges)", 3, 36, 12, 1)

    # Reuse distribution computations from Tab 2 if available
    # (fallback if user hasn't touched Tab 2 yet)
    try:
        ch_df = ch.copy()
        inv = inventory_value
        mp_eval = mp_sorted.copy()
    except NameError:
        landed = procurement_per_kg + ocean_freight_per_kg * float(scenario_row["freight_multiplier"]) + customs_per_kg
        ch_df = pd.DataFrame([
            {"channel":"Home","monthly_volume": 1200*(1+float(scenario_row["demand_shock_pct"])),"price": 14.0*(1+float(scenario_row["price_adj_pct"])),"dist_cost": 1.50},
            {"channel":"Mini","monthly_volume": 1800*(1+float(scenario_row["demand_shock_pct"])),"price": 13.0*(1+float(scenario_row["price_adj_pct"])),"dist_cost": 1.20},
            {"channel":"Restaurant","monthly_volume": 2500*(1+float(scenario_row["demand_shock_pct"])),"price": 12.5*(1+float(scenario_row["price_adj_pct"])),"dist_cost": 1.00},
        ])
        spoilage_pct = float(scenario_row["spoilage_rate_pct"])
        ch_df["revenue"] = ch_df["monthly_volume"] * ch_df["price"]
        ch_df["product_cogs"] = ch_df["monthly_volume"] * landed * (1 + spoilage_pct/100)
        ch_df["distribution_costs"] = ch_df["monthly_volume"] * ch_df["dist_cost"]
        ch_df["gross_profit"] = ch_df["revenue"] - ch_df["product_cogs"] - ch_df["distribution_costs"]
        avg_daily_cogs = ch_df["product_cogs"].sum()/30
        inv = avg_daily_cogs * (int(scenario_row["procurement_cycle_days"]) / 2)

        mp_eval = meating_points.copy()
        mp_eval["base_margin_per_kg"] = (mp_eval["price_per_kg"] - (landed + mp_eval["dist_cost_per_kg"])).clip(lower=0.0)
        uplift = float(fridge_econ["uplift_demand_pct"]) / 100.0
        spoilage_reduction = float(fridge_econ["spoilage_reduction_pct"]) / 100.0
        base_spoilage = spoilage_pct/100.0
        monthly_base_sales = mp_eval["daily_demand_kg"] * 30
        mp_eval["monthly_margin_gain"] = (monthly_base_sales * uplift * mp_eval["base_margin_per_kg"]) + \
                                         (monthly_base_sales * mp_eval["base_margin_per_kg"] * base_spoilage * spoilage_reduction)
        monthly_costs = (float(fridge_econ["capex_sgd"]) / float(fridge_econ["depreciation_months"])) + \
                        float(fridge_econ["maintenance_sgd_month"]) + float(fridge_econ["energy_sgd_month"]) - \
                        float(fridge_econ["rental_income_sgd_month"])
        mp_eval["net_monthly_gain"] = mp_eval["monthly_margin_gain"] - monthly_costs
        mp_eval["payback_months"] = np.where(mp_eval["net_monthly_gain"]>0, float(fridge_econ["capex_sgd"]) / mp_eval["net_monthly_gain"], np.nan)
        mp_eval = mp_eval.sort_values("payback_months")

    gp_margin_val = (ch_df["gross_profit"].sum() / max(ch_df["revenue"].sum(),1))

    recommendations = []

    # Rule 1: Margin below target -> propose lever mix
    if gp_margin_val < target_margin:
        gap = (target_margin - gp_margin_val) * 100
        recommendations.append({
            "priority": "High",
            "action": f"Increase blended price by 2–4% and renegotiate distribution costs (-0.10/kg). Margin gap {gap:.1f}%."
        })

    # Rule 2: Inventory heavy vs sales
    sales = ch_df["revenue"].sum()
    if inv > 0.25 * sales:
        recommendations.append({
            "priority": "Medium",
            "action": "Inventory value >25% of monthly sales: tighten procurement cycle by 3–5 days; implement weekly S&OP."
        })

    # Rule 3: Fridge ROI (deploy where payback ≤ threshold)
    good = mp_eval[mp_eval["payback_months"] <= max_payback_months].head(5)
    if not good.empty:
        ids = ", ".join(good["mp_id"].tolist())
        recommendations.append({
            "priority": "High",
            "action": f"Deploy {len(good)} fridges immediately at: {ids}. All payback ≤ {max_payback_months} months."
        })
    else:
        recommendations.append({
            "priority": "Low",
            "action": "No meating point meets ROI threshold this month. Re-evaluate next cycle after price/volume changes."
        })

    # Rule 4: Logistics stress
    if scenario in ["High Freight","Flood Disruption"]:
        recommendations.append({
            "priority": "High" if scenario=="Flood Disruption" else "Medium",
            "action": f"Scenario ‘{scenario}’: pre-book ocean freight; split shipments; consider temporary price uplift."
        })

    st.markdown("#### Recommendations")
    rec_df = pd.DataFrame(recommendations)
    st.dataframe(rec_df, use_container_width=True, height=240)

    st.markdown("#### Auto-Generated Tasks (ACT)")
    tasks = []
    for i, r in rec_df.iterrows():
        tasks.append({
            "task_id": f"T-{100+i}",
            "priority": r["priority"],
            "owner": np.random.choice(["Trading","Logistics","Sales","Finance"]),
            "task": r["action"]
        })
    st.dataframe(pd.DataFrame(tasks), use_container_width=True, height=220)

    st.caption("This loop illustrates Sense→Plan→Act→Learn. Swap in your LLM to generate narratives while these tools compute ROI, margins, and constraints.")

st.markdown("---")
st.markdown("**About**: This demo visualizes Aliyah Rizq’s core trading workflow and the upcoming distribution business (‘Meating Points’). It simulates an Agentic AI loop using rules over live inputs and scenario toggles. Replace the heuristics with your LLM/LAM for production.")
