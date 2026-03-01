import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Formwork Kitting & BoQ Optimizer",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 Formwork Kitting & BoQ Optimization")
st.write(
    "This web app uses machine learning models trained on synthetic data to estimate "
    "**formwork area**, **cost**, **BoQ adjustments**, and **weekly demand forecasts**."
)
st.markdown("---")


# -------------------------------------------------
# HELPER: SAFE MODEL LOADING
# -------------------------------------------------
@st.cache_resource
def load_model_safe(path_list):
    """
    path_list: list of filenames in priority order.
    Returns (model, used_path) or (None, None) if not found.
    """
    for p in path_list:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m, p
            except Exception as e:
                st.error(f"Error loading model {p}: {e}")
                return None, None
    return None, None


# Load models
cost_model, cost_model_path = load_model_safe(["cost_model_tuned.pkl", "cost_model.pkl"])
boq_model, _ = load_model_safe(["boq_adjustment_model.pkl"])
inv_model, _ = load_model_safe(["inventory_weekly_demand_model.pkl"])


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["1️⃣ Element Estimator", "2️⃣ BoQ Optimizer", "3️⃣ Weekly Demand Forecast"]
)

# =========================================================
# TAB 1: ELEMENT ESTIMATOR
# =========================================================
with tab1:
    st.subheader("1️⃣ Element Estimator – Area & Cost")

    # st.write(
    #     "Yahan tum ek typical formwork element (column/beam/slab/wall) "
    #     "ke dimensions daaloge, aur app: \n"
    #     "- theoretical **formwork area** calculate karega, "
    #     "- cost model se **estimated cost** batayega."
    # )

    # Form for inputs
    with st.form("element_estimator_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            element_type = st.selectbox(
                "Element type",
                ["Column", "Beam", "Slab", "Wall"],
            )
            length_m = st.number_input(
                "Length (m)",
                min_value=0.1,
                max_value=50.0,
                value=0.30,
                step=0.05,
            )
        with c2:
            width_m = st.number_input(
                "Width (m)",
                min_value=0.1,
                max_value=50.0,
                value=0.45,
                step=0.05,
            )
            height_m = st.number_input(
                "Height (m)",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
            )
        with c3:
            quantity_elements = st.number_input(
                "Quantity (identical elements)",
                min_value=1,
                max_value=1000,
                value=20,
                step=1,
            )
            wastage_percent = st.number_input(
                "Wastage (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
            )

        unit_cost_per_sqm = st.number_input(
            "Unit cost per sqm (₹)",
            min_value=100.0,
            max_value=10000.0,
            value=950.0,
            step=10.0,
        )

        submitted_1 = st.form_submit_button("🔮 Estimate Area & Cost")

    if submitted_1:
        # Theoretical area (engineering formula)
        formwork_area_per_element = 2 * (length_m + width_m) * height_m
        total_area_theoretical = (
            formwork_area_per_element
            * quantity_elements
            * (1 + wastage_percent / 100.0)
        )

        # Default: simple cost = area * unit cost
        simple_cost = total_area_theoretical * unit_cost_per_sqm

        # ML cost model prediction (if available)
        ml_cost = None
        if cost_model is not None:
            feature_vec = np.array(
                [
                    [
                        length_m,
                        width_m,
                        height_m,
                        quantity_elements,
                        wastage_percent,
                        unit_cost_per_sqm,
                        total_area_theoretical,
                    ]
                ]
            )
            ml_cost = float(cost_model.predict(feature_vec)[0])

        st.markdown("### 📊 Results")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                "Theoretical Total Formwork Area",
                f"{total_area_theoretical:,.2f} m²",
            )
            st.metric(
                "Simple Cost Estimate (Area × Unit cost)",
                f"₹ {simple_cost:,.0f}",
            )

        with colB:
            if ml_cost is not None:
                st.metric(
                    f"ML-based Cost Estimate (model: {os.path.basename(cost_model_path)})",
                    f"₹ {ml_cost:,.0f}",
                )
                diff = ml_cost - simple_cost
                st.write(
                    f"Difference vs simple estimate: **₹ {diff:,.0f}** "
                    "(+ve → model predicts higher cost, -ve → lower)."
                )
            else:
                st.warning(
                    "Cost model (.pkl) nahin mila. Filhaal sirf simple area × cost estimate dikhaya ja raha hai."
                )

        st.markdown(
            """
            **Interpretation:**
            - The area-based formula provides a deterministic calculation of the required quantity. 
            - The ML-based cost model captures indirect factors such as overheads, material wastage variability, and site-specific uncertainties.  
            - The difference between the two estimates can be leveraged as a risk buffer for contingency allocation and more robust budget planning.
            """
        )

# =========================================================
# TAB 2: BOQ OPTIMIZER
# =========================================================
with tab2:
    st.subheader("2️⃣ BoQ Optimizer – Planned vs Recommended")

    st.write(
        "Here, you can input the planned formwork area for a specific BoQ item, and"
        "the model will recommend the appropriate adjustment—indicating whether the"
        "estimate is potentially overestimated or underestimated."
    )

    if boq_model is None:
        st.error(
            "BoQ adjustment model (`boq_adjustment_model.pkl`) is not loaded. "
            "First, execute the training script to generate the model file.."
        )
    else:
        with st.form("boq_optimizer_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                element_type_boq = st.selectbox(
                    "Element type", ["Column", "Beam", "Slab", "Wall"]
                )
                material_type_boq = st.selectbox(
                    "Material type", ["Plywood", "Steel"]
                )
                floor = st.number_input(
                    "Floor number",
                    min_value=0,
                    max_value=100,
                    value=3,
                    step=1,
                )
            with c2:
                length_b = st.number_input(
                    "Typical Length (m)",
                    min_value=0.1,
                    max_value=50.0,
                    value=0.30,
                    step=0.05,
                )
                width_b = st.number_input(
                    "Typical Width (m)",
                    min_value=0.1,
                    max_value=50.0,
                    value=0.45,
                    step=0.05,
                )
                height_b = st.number_input(
                    "Typical Height (m)",
                    min_value=0.1,
                    max_value=10.0,
                    value=3.0,
                    step=0.1,
                )
            with c3:
                qty_b = st.number_input(
                    "Quantity (approx elements)",
                    min_value=1,
                    max_value=2000,
                    value=40,
                    step=1,
                )
                wastage_b = st.number_input(
                    "Wastage (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                )

            boq_area = st.number_input(
                "Planned BoQ Area (m²)",
                min_value=1.0,
                max_value=50000.0,
                value=1000.0,
                step=10.0,
            )

            c4, c5 = st.columns(2)
            with c4:
                start_week = st.number_input(
                    "Planned start week",
                    min_value=1,
                    max_value=60,
                    value=10,
                    step=1,
                )
            with c5:
                end_week = st.number_input(
                    "Planned end week",
                    min_value=1,
                    max_value=60,
                    value=14,
                    step=1,
                )

            submitted_2 = st.form_submit_button("📉 Optimize BoQ")

        # IMPORTANT: yeh if form ke baahar, lekin tab2 ke andar hona chahiye
        if submitted_2:
            # 1) Compute theoretical area (same as training logic)
            total_area_theoretical_b = (
                2 * (length_b + width_b)
                * height_b
                * qty_b
                * (1 + wastage_b / 100.0)
            )

            # 2) Get exact feature names that model was trained on
            feature_names = list(boq_model.feature_names_in_)
            row = {name: 0.0 for name in feature_names}

            # Helper: set value only if that feature existed at training time
            def set_if(name: str, value):
                if name in row:
                    row[name] = float(value)

            # 3) Fill numeric features (only if present in trained model)
            set_if("length_m", length_b)
            set_if("width_m", width_b)
            set_if("height_m", height_b)
            set_if("quantity_elements", qty_b)
            set_if("wastage_percent", wastage_b)
            set_if("total_formwork_area_sqm", total_area_theoretical_b)
            set_if("boq_area_sqm", boq_area)
            set_if("floor", floor)
            set_if("planned_start_week", start_week)
            set_if("planned_end_week", end_week)

            # 4) One-hot encode element type to match training dummies
            for et in ["Column", "Beam", "Slab", "Wall"]:
                col_name = f"element_type_{et}"
                if col_name in row:
                    row[col_name] = 1.0 if element_type_boq == et else 0.0

            # 5) One-hot encode material type to match training dummies
            for mt in ["Plywood", "Steel"]:
                col_name = f"material_type_{mt}"
                if col_name in row:
                    row[col_name] = 1.0 if material_type_boq == mt else 0.0

            # 6) Convert dict → DataFrame in the correct column order
            X_boq = pd.DataFrame(
                [[row[f] for f in feature_names]],
                columns=feature_names,
            )

            # 7) Predict adjustment factor from model
            adj_factor = float(boq_model.predict(X_boq)[0])

            recommended_area = boq_area * adj_factor
            delta_area = recommended_area - boq_area
            pct_change = (delta_area / boq_area) * 100.0

            st.markdown("### 📊 BoQ Recommendation")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Planned BoQ Area", f"{boq_area:,.2f} m²")
            with col2:
                st.metric("Recommended Area", f"{recommended_area:,.2f} m²")
            with col3:
                st.metric("Change (%)", f"{pct_change:,.2f} %")

            # 8) Cost impact using cost_model (if available)
            if cost_model is not None:
                assumed_unit_cost = 950.0

                # Feature order for cost model:
                # [length_m, width_m, height_m, quantity_elements,
                #  wastage_percent, unit_cost_per_sqm, total_formwork_area_sqm]
                feat_planned = np.array(
                    [
                        [
                            length_b,
                            width_b,
                            height_b,
                            qty_b,
                            wastage_b,
                            assumed_unit_cost,
                            boq_area,  # treat BoQ area as scenario area
                        ]
                    ]
                )
                feat_reco = feat_planned.copy()
                feat_reco[0, -1] = recommended_area  # replace area with recommended area

                cost_planned = float(cost_model.predict(feat_planned)[0])
                cost_recommended = float(cost_model.predict(feat_reco)[0])
                saving = cost_planned - cost_recommended

                st.markdown("### 💰 Cost Impact (Approx)")

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Planned Cost (approx)", f"₹ {cost_planned:,.0f}")
                with cB:
                    st.metric("Recommended Cost (approx)", f"₹ {cost_recommended:,.0f}")
                with cC:
                    st.metric("Estimated Saving", f"₹ {saving:,.0f}")

            # st.markdown(
            #     """
            #     **Note:** Adjustment factor synthetic dataset se learned hai.  
            #     Real deployment mein is model ko actual project history se fine-tune kiya ja sakta hai.
            #     """
            # )

# =========================================================
# TAB 3: WEEKLY DEMAND FORECAST
# =========================================================
with tab3:
    st.subheader("3️⃣ Weekly Demand Forecast – Inventory Planning")

    st.write(
        "Select a week number, and the model will estimate the total formwork area "
        "required for that week across all project elements."
    )

    if inv_model is None:
        st.error(
            "Inventory demand model (`inventory_weekly_demand_model.pkl`) is not loaded."
        )
    else:
        c1, c2 = st.columns([2, 1])

        with c1:
            week_selected = st.slider("Select week", min_value=1, max_value=60, value=15)

        # Predict for selected week
        demand_selected = float(inv_model.predict(np.array([[week_selected]]))[0])

        # Predict for range of weeks for small chart
        weeks_range = np.arange(1, 61)
        demands_range = inv_model.predict(weeks_range.reshape(-1, 1))

        st.markdown("### 📊 Forecast")

        cA, cB = st.columns(2)
        with cA:
            st.metric(
                f"Predicted demand in week {week_selected}",
                f"{demand_selected:,.2f} m²",
            )
        with cB:
            st.write(
                "Use this value for inventory planning – ensure sufficient formwork "
                "is available by this week."
            )

        st.line_chart(
            pd.DataFrame(
                {"Week": weeks_range, "Total Demand (m²)": demands_range}
            ).set_index("Week")
        )

        st.caption(
            "Chart shows model-predicted total formwork area demand per week "
        )