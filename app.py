import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Facebook Ads Dealer Dashboard", page_icon="📈", layout="wide")


RAW_COLUMNS = {
    "dealer": ["dealer", "ten_dealer", "dealer_name", "name"],
    "objective": ["objective", "campaign_objective", "ad_objective", "muc_tieu", "objective_name"],
    "product": ["product", "san_pham", "sku", "item"],
    "campaign_name": [
        "campaign_name",
        "campaign",
        "campaignname",
        "campain_name",
        "campagne",
        "t_n_chi_n_d_ch",
    ],
    "ad_name": ["ad_name", "ad", "adname", "creative_name", "t_n_qu_ng_c_o"],
    "date": ["date", "ngay", "ng_y", "day", "created", "event_date", "report_date"],
    "spend": ["spend", "chi_tieu", "cost", "budget", "s_ti_n_chi_ti_u_vnd"],
    "messages": [
        "messages",
        "message_conversations_started",
        "message_conversations",
        "messages_started",
        "so_luot_tin_nhan",
        "l_t_b_t_u_cu_c_tr_chuy_n_qua_tin_nh_n",
    ],
    "reach": ["reach", "tiep_can", "ng_i_ti_p_c_n"],
    "impressions": ["impressions", "luot_hien_thi", "l_t_hi_n_th"],
    "frequency": ["frequency", "tan_suat", "t_n_su_t"],
}


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("\u200b", "")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def parse_numeric(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.replace(r"\s+", "", regex=True)
    normalized = normalized.where(
        ~normalized.str.contains(r"\.", na=False),
        normalized.str.replace(",", "", regex=False),
    )
    normalized = normalized.where(
        normalized.str.contains(r"\.", na=False),
        normalized.str.replace(",", ".", regex=False),
    )
    cleaned = (
        normalized.str.replace(r"[^\d\.-]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace(r"^$", "Unknown", regex=True)
    )


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def read_csv_with_fallback(file_bytes: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(file_bytes), encoding="cp1252", errors="replace")


def infer_structured_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["dealer"] = df["dealer"].astype(object)
    df["objective"] = df["objective"].astype(object)
    df["product"] = df["product"].astype(object)

    campaign_values = df.get("campaign_name", pd.Series(index=df.index, dtype="object"))
    ad_values = df.get("ad_name", pd.Series(index=df.index, dtype="object"))

    inferred_dealer = []
    inferred_objective = []
    inferred_product = []

    for campaign, ad_name in zip(campaign_values.fillna(""), ad_values.fillna("")):
        campaign_text = str(campaign).strip()
        ad_text = str(ad_name).strip()

        dealer = "Unknown"
        objective = "Unknown"
        product = "Unknown"

        if campaign_text:
            parts = [part.strip() for part in campaign_text.split(" - ")]
            if len(parts) >= 3:
                dealer = parts[1]
                if dealer.lower().startswith("pd x "):
                    dealer = dealer[5:].strip()
                if not dealer:
                    dealer = "Unknown"
                objective = parts[-1] or "Unknown"
            elif len(parts) == 2:
                objective = parts[-1] or "Unknown"

        if ad_text:
            product = ad_text.split(" - ", 1)[0].strip() or "Unknown"

        inferred_dealer.append(dealer)
        inferred_objective.append(objective)
        inferred_product.append(product)

    dealer_mask = df["dealer"].isna() | (df["dealer"].astype(str).str.strip().eq("Unknown"))
    objective_mask = df["objective"].isna() | (df["objective"].astype(str).str.strip().eq("Unknown"))
    product_mask = df["product"].isna() | (df["product"].astype(str).str.strip().eq("Unknown"))

    df.loc[dealer_mask, "dealer"] = pd.Series(inferred_dealer, index=df.index)
    df.loc[objective_mask, "objective"] = pd.Series(inferred_objective, index=df.index)
    df.loc[product_mask, "product"] = pd.Series(inferred_product, index=df.index)

    return df


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()

    if file_bytes.startswith(b"PK"):
        suffix = ".xlsx"

    if suffix == ".csv":
        df = read_csv_with_fallback(file_bytes)
    elif suffix in {".xlsx", ".xlsm"}:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets = xl.sheet_names
        frames = [pd.read_excel(xl, sheet_name=sheet) for sheet in sheets]
        df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel only.")

    df.columns = [normalize_column_name(col) for col in df.columns]

    for canonical, aliases in RAW_COLUMNS.items():
        found = [col for col in df.columns if col in aliases]
        if found:
            df = df.rename(columns={found[0]: canonical})

    df = ensure_required_columns(df)
    df = infer_structured_fields(df)

    for col in ["dealer", "objective", "product", "campaign_name", "ad_name"]:
        df[col] = clean_text(df[col])

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")

    for col in ["spend", "messages", "reach", "impressions", "frequency"]:
        if col in df.columns:
            df[col] = parse_numeric(df[col])

    df["spend"] = df["spend"].fillna(0)
    df["messages"] = df["messages"].fillna(0)
    df["reach"] = df["reach"].fillna(0)
    df["impressions"] = df["impressions"].fillna(0)
    df["frequency"] = df["frequency"].fillna(0)

    mask = (df["frequency"] == 0) & (df["reach"] > 0) & (df["impressions"] > 0)
    df.loc[mask, "frequency"] = df.loc[mask, "impressions"] / df.loc[mask, "reach"]
    df["cost_per_message"] = np.where(df["messages"] > 0, df["spend"] / df["messages"], 0.0)

    df = df[
        [
            "dealer",
            "objective",
            "product",
            "campaign_name",
            "ad_name",
            "date",
            "spend",
            "messages",
            "reach",
            "impressions",
            "frequency",
            "cost_per_message",
        ]
    ]

    return df


def format_currency(value: float) -> str:
    return f"₫ {float(value):,.0f}"


def render_metric_card(title: str, value: str, description: str, theme: str, accent: str = "blue") -> None:
    palette = {
        "blue": ("#0f172a", "#3b82f6", "#93c5fd"),
        "teal": ("#0f172a", "#14b8a6", "#99f6e4"),
        "violet": ("#0f172a", "#8b5cf6", "#ddd6fe"),
        "amber": ("#0f172a", "#f59e0b", "#fde68a"),
        "rose": ("#0f172a", "#f43f5e", "#fecdd3"),
    }
    base, glow, accent_text = palette.get(accent, palette["blue"])
    background = f"linear-gradient(135deg, {base}, {glow})"
    text = "#f8fafc" if theme == "dark" else "#0f172a"
    subtext = accent_text if theme == "dark" else "#334155"
    border = "rgba(255,255,255,0.12)" if theme == "dark" else "rgba(59,130,246,0.18)"

    st.markdown(
        f"""
        <div style="
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            background: {background};
            border: 1px solid {border};
            box-shadow: 0 20px 40px rgba(15,23,42,0.22);
            min-height: 130px;
            position: relative;
            overflow: hidden;
        ">
            <div style="position:absolute; inset:auto -20px -30px auto; width: 90px; height: 90px; border-radius: 999px; background: rgba(255,255,255,0.12);"></div>
            <div style="font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {accent_text}; margin-bottom: 0.55rem;">{title}</div>
            <div style="font-size: 1.85rem; font-weight: 800; color: {text}; margin-bottom: 0.4rem; line-height: 1.1;">{value}</div>
            <div style="font-size: 0.95rem; color: {subtext}; max-width: 18rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame, start_date, end_date, dealer, objective, product, campaign):
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    if dealer != "All":
        df = df[df["dealer"] == dealer]
    if objective != "All":
        df = df[df["objective"] == objective]
    if product != "All":
        df = df[df["product"] == product]
    if campaign != "All":
        df = df[df["campaign_name"] == campaign]
    return df


def build_theme_template(theme: str):
    return "plotly_dark" if theme == "dark" else "plotly"


def style_plotly_fig(fig, font_size=17, legend_font_size=16, grid_color="#e5e7eb", bg_color="#f8fafc", border_radius=18, marker_size=10, line_width=3, height=480):
    fig.update_layout(
        font=dict(family="Inter, Arial, sans-serif", size=font_size, color="#0f172a"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#c7d2fe",
            borderwidth=1,
            font=dict(size=legend_font_size, color="#334155"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=32, r=24, t=48, b=32),
        xaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor="#cbd5e1", linewidth=1, tickfont=dict(size=font_size+2)),
        yaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor="#cbd5e1", linewidth=1, tickfont=dict(size=font_size+2)),
        height=height,
    )
    # Chỉ áp dụng marker size cho line/scatter chart, tránh lỗi với bar chart
    for d in fig.data:
        if getattr(d, "type", "") in {"scatter", "scattergl"} and hasattr(d, "marker"):
            d.marker.size = marker_size
        if getattr(d, "type", "") in {"scatter", "scattergl", "line"} and hasattr(d, "line"):
            d.line.width = line_width
    return fig


def main():
    theme = True
    is_dark = theme
    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 1.1rem;
                padding-bottom: 2.4rem;
            }}
            [data-testid="stSidebar"] {{
                background: {'linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.92))' if theme else 'linear-gradient(180deg, #f8fbff, #eef2ff)'};
                border-right: 1px solid rgba(99, 102, 241, 0.18);
            }}
            [data-testid="stAppViewContainer"] {{
                background: {'radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 24%), radial-gradient(circle at top right, rgba(139,92,246,0.16), transparent 22%), linear-gradient(180deg, #020617 0%, #111827 45%, #0f172a 100%)' if theme else 'radial-gradient(circle at top left, rgba(191,219,254,0.95), transparent 24%), radial-gradient(circle at top right, rgba(224,231,255,0.95), transparent 22%), linear-gradient(180deg, #f8fbff 0%, #eef2ff 42%, #fdf2f8 100%)'};
            }}
            h1, h2, h3 {{
                letter-spacing: -0.03em;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            border-radius: 28px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1.15rem;
            background: {'linear-gradient(135deg, rgba(14,116,144,0.28), rgba(59,130,246,0.22))' if is_dark else 'linear-gradient(135deg, rgba(224,242,254,0.98), rgba(233,213,255,0.96))'};
            border: 1px solid {'rgba(125,211,252,0.18)' if is_dark else 'rgba(99,102,241,0.16)'};
            box-shadow: 0 18px 40px rgba(15,23,42,0.12);
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; gap: 1rem; flex-wrap:wrap;">
                <div>
                    <div style="display:inline-flex; align-items:center; gap:0.45rem; padding:0.35rem 0.75rem; border-radius:999px; background: rgba(59,130,246,0.18); color:#93c5fd; font-size:0.84rem; font-weight:700; margin-bottom: 0.7rem;">
                        ⚡ Live performance intelligence
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: {'#f8fafc' if is_dark else '#0f172a'}; line-height:1.1; margin-bottom: 0.5rem;">
                        Facebook Ads Dealer Analytics Dashboard
                    </div>
                    <div style="font-size: 1rem; color: {'#cbd5e1' if is_dark else '#334155'}; max-width: 56rem;">
                        Theo dõi chi tiêu, hiệu quả tiếp cận, tin nhắn và phân tích dealer trong một không gian trực quan, rõ ràng và đầy màu sắc.
                    </div>
                </div>
                <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
                    <span style="padding:0.45rem 0.8rem; border-radius:999px; background: rgba(45,212,191,0.16); color:#99f6e4; font-weight:700;">📈 KPI chính</span>
                    <span style="padding:0.45rem 0.8rem; border-radius:999px; background: rgba(167,139,250,0.18); color:#ddd6fe; font-weight:700;">🎯 Dealer insight</span>
                    <span style="padding:0.45rem 0.8rem; border-radius:999px; background: rgba(251,191,36,0.18); color:#fde68a; font-weight:700;">🧠 Message performance</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload Excel or CSV file",
        type=["csv", "xlsx", "xlsm"],
        help="Supported files: CSV or Excel (.xlsx/.xlsm)",
    )

    if uploaded is None:
        st.info("Please upload a data file to get started.")
        st.stop()

    with st.spinner("Loading and cleaning your data..."):
        raw_df = load_data(uploaded.getvalue(), uploaded.name)

    if raw_df.empty:
        st.warning("No usable rows found after cleaning. Please verify the file columns and formats.")
        st.stop()

    raw_df = raw_df.dropna(subset=["date"])
    if raw_df.empty:
        st.warning("No rows had a valid date value. Please check the date column format.")
        st.stop()

    min_date = raw_df["date"].min().date()
    max_date = raw_df["date"].max().date()

    with st.sidebar:
        st.header("Filters")
        theme = st.toggle("Dark mode", value=theme, help="Switch between dark and light Plotly themes")
        template = build_theme_template("dark" if theme else "light")
        is_dark = theme

        start_date, end_date = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        dealers = ["All"] + sorted(raw_df["dealer"].unique().tolist())
        objectives = ["All"] + sorted(raw_df["objective"].unique().tolist())
        products = ["All"] + sorted(raw_df["product"].unique().tolist())
        campaigns = ["All"] + sorted(raw_df["campaign_name"].unique().tolist())

        dealer = st.selectbox("Dealer", dealers)
        objective = st.selectbox("Objective", objectives)
        product = st.selectbox("Product", products)
        campaign = st.selectbox("Campaign", campaigns)

        st.divider()
        st.subheader("Navigation")
        view = st.radio(
            "Choose a view",
            [
                "Overview",
                "Dealer Analysis",
                "Product Analysis",
                "Message Performance",
                "Raw Data",
            ],
            horizontal=False,
            label_visibility="collapsed",
            help="Switch between dashboard sections",
        )

    filtered = apply_filters(raw_df, start_date, end_date, dealer, objective, product, campaign)

    if filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    total_spend = float(filtered["spend"].sum())
    total_messages = float(filtered["messages"].sum())
    cost_per_message = total_spend / total_messages if total_messages > 0 else 0.0
    total_reach = float(filtered["reach"].sum())
    total_impressions = float(filtered["impressions"].sum())
    avg_frequency = float(filtered["frequency"].mean()) if len(filtered) else 0.0

    st.subheader("KPI Overview")
    cols = st.columns(6)
    with cols[0]:
        render_metric_card("Total Spend", format_currency(total_spend), "All selected campaigns", "dark" if theme else "light", accent="blue")
    with cols[1]:
        render_metric_card("Messages", f"{total_messages:,.0f}", "Message conversations started", "dark" if theme else "light", accent="violet")
    with cols[2]:
        render_metric_card("Cost per Message", format_currency(cost_per_message), "Spend divided by messages", "dark" if theme else "light", accent="amber")
    with cols[3]:
        render_metric_card("Reach", f"{total_reach:,.0f}", "Unique audience reached", "dark" if theme else "light", accent="teal")
    with cols[4]:
        render_metric_card("Impressions", f"{total_impressions:,.0f}", "Total ad impressions", "dark" if theme else "light", accent="blue")
    with cols[5]:
        render_metric_card("Avg Frequency", f"{avg_frequency:.2f}x", "Average impressions per reach", "dark" if theme else "light", accent="rose")

    if view == "Overview":
        st.subheader("Campaign snapshot")
        overview_cols = st.columns(2)

        with overview_cols[0]:
            fig = px.line(
                filtered.sort_values("date"),
                x="date",
                y="spend",
                color="dealer",
                markers=True,
                title="Spend over time by dealer",
            )
            fig.update_layout(template=template, legend_title_text="Dealer")
            fig = style_plotly_fig(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with overview_cols[1]:
            fig = px.bar(
                filtered.groupby(["dealer", "objective"], as_index=False)["spend"].sum().sort_values("spend", ascending=False),
                x="spend",
                y="dealer",
                color="objective",
                orientation="h",
                title="Spend by dealer and objective",
            )
            fig.update_layout(template=template, legend_title_text="Objective", barmode="group", bargap=0.25)
            fig = style_plotly_fig(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

    elif view == "Dealer Analysis":
        st.subheader("Dealer overview")

        dealer_cols = st.columns(2)
        with dealer_cols[0]:
            fig = px.line(
                filtered.sort_values(["date", "dealer"]),
                x="date",
                y="spend",
                color="dealer",
                markers=True,
                title="Spend by day and dealer",
            )
            fig.update_layout(template=template, legend_title_text="Dealer")
            fig = style_plotly_fig(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with dealer_cols[1]:
            fig = px.line(
                filtered.sort_values(["date", "dealer"]),
                x="date",
                y="messages",
                color="dealer",
                markers=True,
                title="Messages by day and dealer",
            )
            fig.update_layout(template=template, legend_title_text="Dealer")
            fig = style_plotly_fig(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

        dealer_bar_cols = st.columns(2)
        with dealer_bar_cols[0]:
            dealer_cost = (
                filtered.groupby("dealer", as_index=False)[["spend", "messages"]]
                .sum()
                .assign(cost_per_message=lambda x: np.where(x["messages"] > 0, x["spend"] / x["messages"], 0.0))
                .sort_values("cost_per_message", ascending=False)
            )
            fig = px.bar(
                dealer_cost,
                x="dealer",
                y="cost_per_message",
                title="Cost per message by dealer",
                color="dealer",
            )
            fig.update_layout(template=template, height=380, xaxis_title="Dealer", yaxis_title="Cost per message", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with dealer_bar_cols[1]:
            dealer_spend = filtered.groupby("dealer", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
            fig = px.bar(
                dealer_spend,
                x="spend",
                y="dealer",
                orientation="h",
                title="Top dealers by total spend",
                color="dealer",
            )
            fig.update_layout(template=template, height=380, xaxis_title="Spend", yaxis_title="Dealer", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    elif view == "Product Analysis":
        st.subheader("Product analysis")

        product_cols = st.columns(2)
        with product_cols[0]:
            if filtered["dealer"].nunique() <= 6:
                fig = px.bar(
                    filtered.sort_values(["date", "dealer"]),
                    x="date",
                    y="spend",
                    color="product",
                    facet_col="dealer",
                    barmode="relative",
                    title="Spend by product over time per dealer",
                )
                fig.update_layout(template=template, height=380 * max(1, int(np.ceil(filtered["dealer"].nunique() / 2))), legend_title_text="Product")
            else:
                fig = px.bar(
                    filtered.sort_values(["date", "product"]),
                    x="date",
                    y="spend",
                    color="product",
                    title="Spend by product over time",
                )
                fig.update_layout(template=template, height=380, legend_title_text="Product")
            st.plotly_chart(fig, use_container_width=True)

        with product_cols[1]:
            fig = px.bar(
                filtered.groupby(["dealer", "product"], as_index=False)["messages"].sum().sort_values(["dealer", "messages"], ascending=[True, False]),
                x="product",
                y="messages",
                color="dealer",
                barmode="group",
                title="Messages by product and dealer",
            )
            fig.update_layout(template=template, height=380, legend_title_text="Dealer", xaxis_title="Product", yaxis_title="Messages")
            st.plotly_chart(fig, use_container_width=True)

        cpm_table = (
            filtered.groupby(["dealer", "product"], as_index=False)[["spend", "messages"]]
            .sum()
            .assign(cost_per_message=lambda x: np.where(x["messages"] > 0, x["spend"] / x["messages"], 0.0))
            .pivot(index="dealer", columns="product", values="cost_per_message")
        )
        cpm_table = cpm_table.fillna(0)

        st.subheader("Cost per message matrix")
        fig = go.Figure(
            data=go.Heatmap(
                z=cpm_table.values,
                x=cpm_table.columns,
                y=cpm_table.index,
                colorscale="Viridis",
                text=np.round(cpm_table.values, 2),
                texttemplate="%{text}",
                hovertemplate="Dealer: %{y}<br>Product: %{x}<br>Cost / message: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(template=template, height=420, xaxis_title="Product", yaxis_title="Dealer")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Message Performance":
        st.subheader("Message performance")
        perf_cols = st.columns(2)

        with perf_cols[0]:
            top_campaigns = (
                filtered.groupby("campaign_name", as_index=False)["messages"].sum().sort_values("messages", ascending=False).head(12)
            )
            fig = px.bar(
                top_campaigns,
                x="campaign_name",
                y="messages",
                title="Top campaigns by messages",
            )
            fig.update_layout(template=template, height=380, xaxis_title="Campaign", yaxis_title="Messages")
            st.plotly_chart(fig, use_container_width=True)

        with perf_cols[1]:
            objective_perf = (
                filtered.groupby("objective", as_index=False)[["messages", "spend"]]
                .sum()
                .assign(cost_per_message=lambda x: np.where(x["messages"] > 0, x["spend"] / x["messages"], 0.0))
                .sort_values("cost_per_message", ascending=False)
            )
            fig = px.bar(
                objective_perf,
                x="objective",
                y="cost_per_message",
                title="Cost per message by objective",
            )
            fig.update_layout(template=template, height=380, xaxis_title="Objective", yaxis_title="Cost per message")
            st.plotly_chart(fig, use_container_width=True)

        scatter = px.scatter(
            filtered,
            x="messages",
            y="spend",
            color="dealer",
            size="impressions",
            hover_data=["product", "objective", "campaign_name", "ad_name"],
            title="Spend vs messages by dealer",
        )
        scatter.update_layout(template=template, height=380)
        st.plotly_chart(scatter, use_container_width=True)

    else:
        st.subheader("Performance table")
        search_term = st.text_input("Search table", placeholder="Search dealer, objective, product, campaign, or ad name")
        display_df = filtered.copy()
        display_df["cost_per_message"] = display_df["cost_per_message"].replace(0, 0.0)
        if search_term:
            mask = display_df.apply(lambda row: search_term.lower() in " | ".join([str(row[col]) for col in ["dealer", "objective", "product", "campaign_name", "ad_name"]]).lower(), axis=1)
            display_df = display_df.loc[mask]

        display_df = display_df.rename(
            columns={
                "dealer": "Dealer",
                "objective": "Objective",
                "product": "Product",
                "campaign_name": "Campaign",
                "ad_name": "Ad Name",
                "date": "Date",
                "spend": "Spend",
                "messages": "Messages",
                "reach": "Reach",
                "impressions": "Impressions",
                "frequency": "Frequency",
                "cost_per_message": "Cost per Message",
            }
        )
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df["Spend"] = display_df["Spend"].map(format_currency)
        display_df["Cost per Message"] = display_df["Cost per Message"].map(lambda x: format_currency(x))
        display_df["Reach"] = display_df["Reach"].map(lambda x: f"{float(x):,.0f}")
        display_df["Impressions"] = display_df["Impressions"].map(lambda x: f"{float(x):,.0f}")
        display_df["Frequency"] = display_df["Frequency"].map(lambda x: f"{float(x):.2f}x")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export current view to CSV",
            data=csv_data,
            file_name="facebook_ads_filtered_export.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
