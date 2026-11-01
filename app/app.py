"""Credit Portfolio Segmentation: Streamlit dashboard."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).parent / "data"

# Segment names come from the cluster profiles in notebook 02.
# If you re-fit K-Means, check these still match.
CLUSTER_NAMES = {
    0: "Established mid-income",
    1: "Early-career, small loans",
    2: "High income, large loans",
    3: "Prime, low risk",
}

st.set_page_config(page_title="Credit Portfolio Segmentation", layout="wide")
px.defaults.template = "plotly_white"  # white chart background


@st.cache_data
def load(name: str) -> pd.DataFrame:
    """Read one of the small summary CSV files used by the dashboard."""
    return pd.read_csv(DATA / name)


profile = load("cluster_profile.csv")
profile["segment"] = profile["kmeans_cluster"].map(CLUSTER_NAMES)
sample = load("dashboard_sample.csv")
overall_rate = (
    (profile["size"] * profile["default_rate"]).sum() / profile["size"].sum()
)

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Cluster profiles", "Visualizations", "Default analysis",
     "Algorithm comparison", "Anomalies", "Insights & limitations"],
)

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Credit Portfolio Segmentation and Risk Pattern Analysis")
    st.write(
        "Borrowers are grouped by financial profile using unsupervised learning "
        "(K-Means, Hierarchical, DBSCAN). The default status was **not** used to "
        "form the groups; it is only used to compare outcomes afterwards."
    )
    low = profile.loc[profile["default_rate"].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers", f"{int(profile['size'].sum()):,}")
    c2.metric("Segments", len(profile))
    c3.metric("Overall default rate", f"{overall_rate:.1%}")
    c4.metric(
        "Lowest-risk segment",
        low["segment"],
        f"{low['default_rate']:.1%} default rate",
        delta_color="off",
    )
    fig = px.pie(
        profile, names="segment", values="size", title="Portfolio share by segment"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------- Cluster profiles
elif page == "Cluster profiles":
    st.title("Cluster profiles")
    table = profile.rename(columns={
        "size": "Borrowers", "revenue": "Avg income", "dti": "Avg DTI",
        "loan_amnt": "Avg loan", "fico": "Avg FICO",
        "emp_years": "Avg years employed", "default_rate": "Default rate",
    }).drop(columns="kmeans_cluster").set_index("segment")
    st.dataframe(table, use_container_width=True)

    feature = st.selectbox(
        "Compare segments by",
        ["revenue", "dti", "loan_amnt", "fico", "emp_years", "default_rate"],
    )
    st.plotly_chart(
        px.bar(profile, x="segment", y=feature, color="segment",
               title=f"Average {feature} by segment"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    purpose = load("cluster_purpose.csv")
    purpose["segment"] = purpose["kmeans_cluster"].map(CLUSTER_NAMES)
    purpose_long = purpose.drop(columns="kmeans_cluster").melt(
        id_vars="segment", var_name="purpose", value_name="share")
    col1.plotly_chart(
        px.bar(purpose_long, x="segment", y="share", color="purpose",
               title="Loan purpose mix"),
        use_container_width=True,
    )
    own = load("cluster_ownership.csv")
    own["segment"] = own["kmeans_cluster"].map(CLUSTER_NAMES)
    own_long = own.drop(columns="kmeans_cluster").melt(
        id_vars="segment", var_name="home ownership", value_name="share")
    col2.plotly_chart(
        px.bar(own_long, x="segment", y="share", color="home ownership",
               title="Home ownership mix"),
        use_container_width=True,
    )

# ----------------------------------------------------------- Visualizations
elif page == "Visualizations":
    st.title("Borrower map (10,000-borrower sample)")
    method = st.radio("Projection", ["PCA", "t-SNE", "UMAP"], horizontal=True)
    algo = st.radio(
        "Color by", ["K-Means", "Hierarchical", "DBSCAN"], horizontal=True
    )
    cols = {
        "PCA": ("pca_1", "pca_2"),
        "t-SNE": ("tsne_1", "tsne_2"),
        "UMAP": ("umap_1", "umap_2"),
    }
    color_col = {
        "K-Means": "kmeans_cluster",
        "Hierarchical": "hier_cluster",
        "DBSCAN": "dbscan_cluster",
    }[algo]

    plot_df = sample.copy()
    plot_df["cluster"] = plot_df[color_col].astype(str)
    x, y = cols[method]
    fig = px.scatter(plot_df, x=x, y=y, color="cluster", opacity=0.6,
                     hover_data=["fico_n", "revenue", "loan_amnt", "dti_n"],
                     title=f"{method} projection, colored by {algo}")
    fig.update_traces(marker={"size": 4})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The groups overlap because borrower traits change gradually; "
               "there are no sharp boundaries between segments.")

# --------------------------------------------------------- Default analysis
elif page == "Default analysis":
    st.title("Historical default patterns")
    fig = px.bar(profile, x="segment", y="default_rate", color="segment",
                 title="Default rate by segment")
    fig.add_hline(y=overall_rate, line_dash="dash", line_color="red",
                  annotation_text="Overall")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Default rate by state")
    state = load("state_default.csv")
    min_n = st.slider("Minimum loans per state", 0, 20000, 1000, step=500)
    state = state[state["size"] >= min_n].sort_values("mean", ascending=False)
    st.plotly_chart(
        px.bar(state, x="addr_state", y="mean", title="Default rate by state",
               labels={"mean": "Default rate", "addr_state": "State"}),
        use_container_width=True,
    )

    st.subheader("Trend by issue year")
    year = load("year_cluster_default.csv").melt(
        id_vars="issue_year", var_name="cluster", value_name="default_rate")
    year["segment"] = year["cluster"].astype(int).map(CLUSTER_NAMES)
    st.plotly_chart(
        px.line(year, x="issue_year", y="default_rate", color="segment",
                markers=True),
        use_container_width=True,
    )

# ------------------------------------------------------ Algorithm comparison
elif page == "Algorithm comparison":
    st.title("Clustering algorithm comparison")
    st.dataframe(load("algorithm_comparison.csv"), use_container_width=True)
    st.markdown(
        "- **K-Means** gave the highest silhouette and clear business segments.\n"
        "- **Hierarchical (Ward)** produced a similar low-risk group.\n"
        "- **DBSCAN** found one dense core, so it is used for anomaly detection "
        "(its silhouette is undefined with one cluster)."
    )

# ---------------------------------------------------------------- Anomalies
elif page == "Anomalies":
    st.title("Anomalies found by DBSCAN")
    noise = sample[sample["dbscan_cluster"] == -1]
    normal = sample[sample["dbscan_cluster"] != -1]
    st.metric("Anomalous borrowers (sample)", f"{len(noise):,}",
              f"{len(noise) / len(sample):.1%} of sample")

    feats = ["revenue", "dti_n", "loan_amnt", "fico_n", "emp_length_years", "Default"]
    cmp_df = pd.DataFrame({
        "Anomalies": noise[feats].mean(),
        "Typical": normal[feats].mean(),
    }).round(2)
    st.dataframe(cmp_df, use_container_width=True)

    plot_df = sample.copy()
    plot_df["type"] = plot_df["dbscan_cluster"].map(
        lambda v: "Anomaly" if v == -1 else "Typical"
    )
    st.plotly_chart(
        px.scatter(plot_df, x="fico_n", y="revenue", color="type", opacity=0.6,
                   title="FICO vs income: anomalies vs typical borrowers"),
        use_container_width=True,
    )
    st.info("These anomalies are unusual, not necessarily risky: they tend to have "
            "very high FICO scores, higher income and larger loans.")

# -------------------------------------------------- Insights and limitations
else:
    st.title("Business insights and limitations")
    st.subheader("Insights")
    st.markdown(
        "- One segment (high FICO, low DTI) has roughly **half the overall "
        "default rate**.\n"
        "- The other segments have similar default rates, so credit score and DTI "
        "separate risk better than income or loan size.\n"
        "- Rent/mortgage/own mix and loan purpose differ across segments and can "
        "guide product design.\n"
        "- Default rates by state and by issue year support geographic and trend "
        "monitoring."
    )
    st.subheader("Limitations")
    st.markdown(
        "- Silhouette scores are low (about 0.2): borrower traits form a continuum.\n"
        "- t-SNE, UMAP, hierarchical clustering and DBSCAN were run on a "
        "10,000-row sample.\n"
        "- Segments describe **groups**, not individuals; do not draw conclusions "
        "about a single borrower from their cluster.\n"
        "- Historical default rates do not guarantee future outcomes."
    )