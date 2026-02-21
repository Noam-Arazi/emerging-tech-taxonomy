"""
Taxonomy Classification UI
===========================
Streamlit-based local interface for classifying emerging technologies.
Upload an Excel/CSV file → classify → explore results interactively.

Usage:
    streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px

import hierarchical_taxonomy as ht
from hierarchical_taxonomy import (
    create_bedrock_client,
    compose_text,
    get_embeddings,
    correction_loop,
    run_stage3,
    reassign_orphans,
    merge_small_leaves,
    name_all_clusters_bottom_up,
    build_result_dataframe,
    assign_ids,
    build_summary,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Taxonomy Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SIDEBAR — AWS CREDENTIALS
# ============================================================================

st.sidebar.title("AWS Configuration")

default_key = ""
default_secret = ""
default_region = "us-east-1"
try:
    from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    default_key = AWS_ACCESS_KEY_ID
    default_secret = AWS_SECRET_ACCESS_KEY
    default_region = AWS_REGION
    st.sidebar.success("Credentials loaded from config.py")
except ImportError:
    st.sidebar.info("Enter AWS credentials below (or create config.py)")

aws_key    = st.sidebar.text_input("AWS Access Key ID",     value=default_key,    type="password")
aws_secret = st.sidebar.text_input("AWS Secret Access Key", value=default_secret, type="password")
aws_region = st.sidebar.text_input("AWS Region",            value=default_region)

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("Emerging Technology Taxonomy Classification")
st.markdown("Upload an Excel/CSV file with technologies → get a hierarchical classification")

# ============================================================================
# FILE UPLOAD
# ============================================================================

uploaded_file = st.file_uploader(
    "Drag & drop your technology file here",
    type=["xlsx", "xls", "csv"],
    help="Required columns: technology_name, Technology_Description",
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"Loaded **{len(df)} technologies** from `{uploaded_file.name}`")

    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    required = ["technology_name", "Technology_Description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    optional = ["Web_of_Science_Category", "OECD_Research_Area",
                "emerging_tech_concepts", "applied_domain"]
    present = [c for c in optional if c in df.columns]
    if present:
        st.info(f"Enrichment columns found: {', '.join(present)}")

    # ========================================================================
    # CLASSIFY BUTTON
    # ========================================================================

    if st.button("Classify Technologies", type="primary", use_container_width=True):
        if not aws_key or not aws_secret:
            st.error("Please enter AWS credentials in the sidebar.")
            st.stop()

        # Override module-level credentials so create_bedrock_client() picks them up
        ht.AWS_ACCESS_KEY_ID     = aws_key
        ht.AWS_SECRET_ACCESS_KEY = aws_secret
        ht.AWS_REGION            = aws_region

        progress = st.progress(0)
        status   = st.empty()

        try:
            # Step 1 — Embeddings
            status.info("Step 1 / 5 — Connecting to AWS Bedrock & generating embeddings…")
            progress.progress(5)
            client = create_bedrock_client()
            texts  = [compose_text(row) for _, row in df.iterrows()]
            embeddings = get_embeddings(texts, client)
            progress.progress(20)

            # Step 2 — Correction loop (tag metrics + LLM coherence + Stage 2)
            status.info("Step 2 / 5 — Selecting best clustering (tag metrics + LLM coherence)…")
            result, quality = correction_loop(df, embeddings, client)
            if result is None:
                st.error("No valid clustering found. Try with different data.")
                st.stop()
            l1_labels, l1_names, l2_structure, l1_desc = result
            progress.progress(55)

            # Step 3 — Optional L3 splits
            status.info("Step 3 / 5 — Optional L3 sub-splits…")
            l3_structure = run_stage3(df, embeddings, l1_labels, l1_names, l2_structure, client)
            progress.progress(65)

            # Step 4 — Post-clustering refinements
            status.info("Step 4 / 5 — Orphan reassignment & small leaf merging…")
            l2_structure, l3_structure = reassign_orphans(
                df, embeddings, l1_labels, l2_structure, l3_structure
            )
            l2_structure, l3_structure = merge_small_leaves(
                df, embeddings, l1_labels, l1_names, l2_structure, l3_structure, client
            )
            progress.progress(75)

            # Step 5 — Bottom-up naming (identical to main())
            status.info("Step 5 / 5 — Naming clusters bottom-up…")
            l1_names, l1_summaries, l2_structure, l3_structure = name_all_clusters_bottom_up(
                df, l1_labels, l1_names, l2_structure, l3_structure, client
            )

            summaries = {}
            for l1_id, name in l1_names.items():
                summaries[name] = l1_summaries.get(l1_id, "")
            for l1_id, l2_groups in l2_structure.items():
                for l2_id, l2_info in l2_groups.items():
                    summaries[l2_info["name"]] = l2_info.get("summary", "")
            for (l1_id, l2_id), l3_groups in l3_structure.items():
                for l3_id, l3_info in l3_groups.items():
                    summaries[l3_info["name"]] = l3_info.get("summary", "")

            result_df  = build_result_dataframe(df, l1_labels, l1_names, l2_structure, l3_structure)
            result_df  = assign_ids(result_df)
            summary_df = build_summary(result_df, summaries=summaries)

            progress.progress(100)
            status.success(f"Done! Algorithm: {l1_desc} | Quality: {quality:.3f}")

            st.session_state["result_df"]  = result_df
            st.session_state["summary_df"] = summary_df
            st.session_state["summaries"]  = summaries
            st.session_state["l1_desc"]    = l1_desc
            st.session_state["quality"]    = quality
            st.session_state["nav_path"]   = []

        except Exception as e:
            st.error(f"Error during classification: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# RESULTS
# ============================================================================

if "result_df" not in st.session_state:
    st.stop()

result_df  = st.session_state["result_df"]
summary_df = st.session_state["summary_df"]
summaries  = st.session_state.get("summaries", {})

st.divider()
st.header("Results")

# ── Metrics ──────────────────────────────────────────────────────────────────

leaf_series     = result_df.groupby("leaf_category").size()
n_l1            = result_df["category_level_1"].nunique()
n_leaves        = len(leaf_series)
avg_per_leaf    = leaf_series.mean()
median_per_leaf = leaf_series.median()
max_l1_pct      = result_df["category_level_1"].value_counts().iloc[0] / len(result_df) * 100
has_l3          = result_df["category_level_3"].ne("").any()
depth           = 3 if has_l3 else 2

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("L1 Groups",       n_l1)
c2.metric("Leaf Categories", n_leaves)
c3.metric("Avg / Leaf",      f"{avg_per_leaf:.1f}")
c4.metric("Median / Leaf",   f"{median_per_leaf:.0f}")
c5.metric("Largest L1",      f"{max_l1_pct:.1f}%")
c6.metric("Tree Depth",      depth)

with st.expander("Leaf size details"):
    d1, d2, d3 = st.columns(3)
    d1.metric("Smallest leaf",  leaf_series.min())
    d2.metric("Largest leaf",   leaf_series.max())
    d3.metric("Leaves ≤ 3 techs", int((leaf_series <= 3).sum()))

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_tree, tab_sunburst, tab_table, tab_search, tab_download = st.tabs([
    "🌲 Tree Navigator", "🌀 Sunburst", "📋 Full Table", "🔍 Search", "⬇️ Download"
])

# ============================================================================
# TAB 1 — TREE NAVIGATOR
# ============================================================================

with tab_tree:
    if "nav_path" not in st.session_state:
        st.session_state["nav_path"] = []

    nav_path = st.session_state["nav_path"]

    # ── Breadcrumb ────────────────────────────────────────────────────────────
    breadcrumb_parts = ["🏠 All groups"] + nav_path
    crumb_cols = st.columns(len(breadcrumb_parts) + 1)   # +1 for spacer

    if crumb_cols[0].button("🏠 All groups", key="bc_root", use_container_width=False):
        st.session_state["nav_path"] = []
        st.rerun()

    for i, name in enumerate(nav_path):
        label = (name[:18] + "…") if len(name) > 18 else name
        if crumb_cols[i + 1].button(f"▸ {label}", key=f"bc_{i}"):
            st.session_state["nav_path"] = nav_path[: i + 1]
            st.rerun()

    st.markdown("---")

    # ── Current level: find children ─────────────────────────────────────────
    current_parent = nav_path[-1] if nav_path else "-"
    children = summary_df[summary_df["parent"] == current_parent].copy()

    if len(children) > 0:
        # ── Show sub-group cards ──────────────────────────────────────────────
        for _, row in children.iterrows():
            cat_name    = row["category"]
            tech_count  = int(row["tech_count"])
            cat_id      = str(row.get("id", ""))
            summary_txt = str(row.get("summary", ""))
            has_children = cat_name in summary_df["parent"].values

            with st.container(border=True):
                left, right = st.columns([9, 1])
                with left:
                    st.markdown(
                        f"**{cat_id} &nbsp; {cat_name}** &nbsp;&nbsp; "
                        f"`{tech_count} {'technology' if tech_count == 1 else 'technologies'}`"
                    )
                    if summary_txt and summary_txt != "nan":
                        st.caption(summary_txt)
                with right:
                    btn_label = "Explore →" if has_children else "View →"
                    if st.button(btn_label, key=f"nav_{cat_name}"):
                        st.session_state["nav_path"] = nav_path + [cat_name]
                        st.rerun()

    else:
        # ── Leaf node — show technology list ──────────────────────────────────
        current_cat = nav_path[-1] if nav_path else None
        if current_cat:
            leaf_techs = (
                result_df[result_df["leaf_category"] == current_cat]
                [["technology_name", "Technology_Description"]]
                .reset_index(drop=True)
            )
            leaf_techs.index += 1

            st.markdown(f"**{len(leaf_techs)} technologies in this group:**")
            st.dataframe(
                leaf_techs.rename(columns={
                    "technology_name":        "Technology",
                    "Technology_Description": "Description",
                }),
                use_container_width=True,
                height=min(500, 80 + len(leaf_techs) * 38),
            )

# ============================================================================
# TAB 2 — SUNBURST
# ============================================================================

with tab_sunburst:
    st.subheader("Hierarchy Overview")

    sun_rows = []
    for _, row in result_df.iterrows():
        l1 = row["category_level_1"]
        l2 = row["category_level_2"] if row["category_level_2"] else l1
        l3 = row["category_level_3"] if row["category_level_3"] else row["leaf_category"]
        sun_rows.append({"L1": l1, "L2": l2, "L3": l3})
    sun_df = pd.DataFrame(sun_rows)

    fig = px.sunburst(sun_df, path=["L1", "L2", "L3"], height=700)
    fig.update_traces(textinfo="label+value")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3 — FULL TABLE
# ============================================================================

with tab_table:
    selected_l1 = st.multiselect(
        "Filter by L1 group:",
        options=sorted(result_df["category_level_1"].unique()),
    )
    display_df = (
        result_df if not selected_l1
        else result_df[result_df["category_level_1"].isin(selected_l1)]
    )
    display_cols = [
        "full_numeric_id", "technology_name",
        "category_level_1", "category_level_2", "category_level_3",
        "leaf_category",
    ]
    available = [c for c in display_cols if c in display_df.columns]
    st.dataframe(
        display_df[available].sort_values("full_numeric_id"),
        use_container_width=True,
        height=600,
    )

# ============================================================================
# TAB 4 — SEARCH
# ============================================================================

with tab_search:
    query = st.text_input("Search technology name:", placeholder="e.g. DNA battery")
    if query:
        mask = result_df["technology_name"].str.contains(query, case=False, na=False)
        hits = result_df[mask]
        if len(hits) == 0:
            st.info("No technologies found.")
        else:
            st.markdown(f"**{len(hits)} result(s):**")
            for _, hit in hits.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{hit['technology_name']}**")
                    st.caption(f"📍 {hit['full_path']}")
                    desc = str(hit.get("Technology_Description", ""))
                    st.write(desc[:300] + ("…" if len(desc) > 300 else ""))

# ============================================================================
# TAB 5 — DOWNLOAD
# ============================================================================

with tab_download:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer,  sheet_name="Technologies",   index=False)
        summary_df.to_excel(writer, sheet_name="Cluster Summary", index=False)

    st.download_button(
        label="⬇️ Download Excel",
        data=output.getvalue(),
        file_name="hierarchical_taxonomy_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
