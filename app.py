
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
MODEL_PATH = BASE_DIR / "models" / "marketingiq_lightgbm.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

st.set_page_config(
    page_title="MarketingIQ | Marketing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# The model was trained without duration and y.
FEATURES = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "campaign", "pdays", "previous", "poutcome"
]

NUMERIC_FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]

CATEGORICAL_OPTIONS = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ],
    "marital": ["divorced", "married", "single"],
    "education": [
        "primary", "secondary", "tertiary", "unknown"
    ],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ],
    "poutcome": ["failure", "other", "success", "unknown"]
}

# =========================================================
# 2. LOAD MODEL, THRESHOLD AND DATA
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Only load trusted model files created by you.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, sep=";")

    missing = set(FEATURES + ["y"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    return data


@st.cache_data
def load_threshold():
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    threshold = float(config["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    return threshold


try:
    model = load_model()
    df = load_data()
    threshold = load_threshold()
except Exception as exc:
    st.error(f"Could not initialize MarketingIQ: {exc}")
    st.stop()

# Check that the expected input columns match the saved pipeline.
try:
    expected_features = list(model.feature_names_in_)
    if expected_features != FEATURES:
        st.error(
            "The saved model's feature order/names do not match this app. "
            "Check the training notebook before proceeding."
        )
        st.stop()
except AttributeError:
    # Some sklearn Pipeline versions do not expose this attribute.
    pass


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def prepare_input(data):
    """Validate and order features exactly as expected by the model."""
    missing = set(FEATURES) - set(data.columns)

    if missing:
        raise ValueError(
            "Missing required input columns: "
            + ", ".join(sorted(missing))
        )

    result = data[FEATURES].copy()

    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if result[NUMERIC_FEATURES].isna().any().any():
        raise ValueError(
            "Numeric fields contain missing or invalid values."
        )

    for col in CATEGORICAL_FEATURES:
        result[col] = result[col].fillna("unknown").astype(str)

    return result


def predict_customers(data):
    """Return probability and threshold-based classification."""
    clean_data = prepare_input(data)

    probabilities = model.predict_proba(clean_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()
    results["subscription_probability"] = probabilities
    results["predicted_subscription"] = np.where(
        predictions == 1, "Likely to subscribe", "Less likely to subscribe"
    )

    return results


def probability_label(probability):
    if probability >= threshold:
        return "Likely to subscribe"
    return "Less likely to subscribe"


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 MarketingIQ")
st.sidebar.caption("Marketing intelligence and response prediction")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customer Prediction",
        "Batch Prediction",
        "Customer Analytics",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.metric("Decision threshold", f"{threshold:.2f}")
st.sidebar.caption(
    "Predictions are estimates from a historical dataset, "
    "not guarantees of customer behavior."
)


# =========================================================
# 5. OVERVIEW
# =========================================================

if page == "Overview":
    st.title("MarketingIQ Dashboard")
    st.write(
        "Understand historical marketing responses and estimate "
        "which customers may subscribe."
    )

    total_customers = len(df)
    subscribers = int((df["y"] == "yes").sum())
    non_subscribers = int((df["y"] == "no").sum())
    response_rate = subscribers / total_customers * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total customers", f"{total_customers:,}")
    c2.metric("Historical subscribers", f"{subscribers:,}")
    c3.metric("Non-subscribers", f"{non_subscribers:,}")
    c4.metric("Historical response rate", f"{response_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Subscription distribution")

        target_counts = (
            df["y"].value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            target_counts,
            names="Outcome",
            values="Customers",
            hole=0.55,
            title="Historical outcomes"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Response by contact method")

        contact_data = (
            df.groupby("contact", observed=True)["y"]
            .apply(lambda values: (values == "yes").mean() * 100)
            .reset_index(name="Response rate (%)")
            .sort_values("Response rate (%)", ascending=False)
        )

        fig = px.bar(
            contact_data,
            x="contact",
            y="Response rate (%)",
            title="Historical response rate by contact method",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical response by age group")

    age_data = df.copy()
    age_data["age_group"] = pd.cut(
        age_data["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
    )

    age_summary = (
        age_data.groupby("age_group", observed=True)["y"]
        .apply(lambda values: (values == "yes").mean() * 100)
        .reset_index(name="Response rate (%)")
    )

    fig = px.bar(
        age_summary,
        x="age_group",
        y="Response rate (%)",
        title="Historical subscription response by age group",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Historical patterns describe this dataset. They do not establish "
        "that a contact method or customer attribute causes a higher "
        "subscription rate."
    )


# =========================================================
# 6. SINGLE-CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":
    st.title("Customer Subscription Prediction")
    st.write(
        "Enter the customer's profile and campaign history to estimate "
        "the probability of subscription."
    )

    st.warning(
        "The model uses the historical Bank Marketing dataset. "
        "Some fields, such as campaign count and previous campaign outcome, "
        "must be known at the time of prediction. Do not enter information "
        "that would only become available after the prediction decision."
    )

    with st.form("customer_form"):
        st.subheader("Customer profile")

        a, b, c = st.columns(3)

        with a:
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=35
            )
            job = st.selectbox("Job", CATEGORICAL_OPTIONS["job"])
            marital = st.selectbox(
                "Marital status", CATEGORICAL_OPTIONS["marital"]
            )

        with b:
            education = st.selectbox(
                "Education", CATEGORICAL_OPTIONS["education"]
            )
            balance = st.number_input(
                "Account balance", value=1000, step=100
            )
            default = st.selectbox(
                "Credit in default?", CATEGORICAL_OPTIONS["default"]
            )

        with c:
            housing = st.selectbox(
                "Housing loan?", CATEGORICAL_OPTIONS["housing"]
            )
            loan = st.selectbox(
                "Personal loan?", CATEGORICAL_OPTIONS["loan"]
            )
            contact = st.selectbox(
                "Contact method", CATEGORICAL_OPTIONS["contact"]
            )

        st.subheader("Campaign information")

        a, b, c = st.columns(3)

        with a:
            day = st.number_input(
                "Last contact day of month",
                min_value=1, max_value=31, value=15
            )
            month = st.selectbox(
                "Last contact month", CATEGORICAL_OPTIONS["month"]
            )

        with b:
            campaign = st.number_input(
                "Contacts in this campaign",
                min_value=1, max_value=100, value=1
            )
            pdays = st.number_input(
                "Days since previous contact (-1 = never contacted)",
                min_value=-1, max_value=999, value=-1
            )

        with c:
            previous = st.number_input(
                "Contacts before this campaign",
                min_value=0, max_value=100, value=0
            )
            poutcome = st.selectbox(
                "Previous campaign outcome",
                CATEGORICAL_OPTIONS["poutcome"]
            )

        submitted = st.form_submit_button(
            "Predict subscription probability",
            type="primary",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        try:
            result = predict_customers(customer)
            probability = float(
                result["subscription_probability"].iloc[0]
            )
            predicted_class = probability_label(probability)

            st.divider()
            st.subheader("Prediction result")

            m1, m2 = st.columns(2)

            m1.metric("Estimated subscription probability",
                      f"{probability:.1%}")
            m2.metric("Model classification", predicted_class)

            st.progress(float(np.clip(probability, 0, 1)))

            if probability >= threshold:
                st.success(
                    "The estimated probability meets or exceeds the "
                    "configured classification threshold."
                )
            else:
                st.info(
                    "The estimated probability is below the configured "
                    "classification threshold."
                )

            st.caption(
                "This is a model score, not a guarantee or a calibrated "
                "business forecast. Validate probabilities before using "
                "them for consequential decisions."
            )

            st.dataframe(result, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =========================================================
# 7. BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":
    st.title("Batch Customer Prediction")
    st.write(
        "Upload a CSV containing the 15 model input columns to score "
        "multiple customers."
    )

    st.info(
        "Required columns: " + ", ".join(FEATURES)
    )

    template = pd.DataFrame([{
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1000,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }])

    st.download_button(
        "Download sample input CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="marketingiq_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.write("Preview of uploaded data")
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            if len(uploaded_df) == 0:
                st.warning("The uploaded CSV contains no customer records.")
            elif len(uploaded_df) > 100000:
                st.error(
                    "Please upload at most 100,000 rows per file."
                )
            elif st.button("Generate predictions", type="primary"):
                result = predict_customers(uploaded_df)

                st.success(
                    f"Generated predictions for {len(result):,} customers."
                )

                total = len(result)
                likely = int(
                    (result["predicted_subscription"] ==
                     "Likely to subscribe").sum()
                )

                x1, x2, x3 = st.columns(3)
                x1.metric("Customers scored", f"{total:,}")
                x2.metric("Predicted likely", f"{likely:,}")
                x3.metric("Predicted likely (%)",
                          f"{likely / total:.1%}" if total else "0%")

                st.dataframe(
                    result.sort_values(
                        "subscription_probability", ascending=False
                    ),
                    use_container_width=True
                )

                st.download_button(
                    "Download prediction results",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="marketingiq_predictions.csv",
                    mime="text/csv"
                )

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")


# =========================================================
# 8. CUSTOMER ANALYTICS
# =========================================================

elif page == "Customer Analytics":
    st.title("Customer Analytics")
    st.write("Explore historical customer patterns in the dataset.")

    left, right = st.columns(2)

    with left:
        selected_job = st.selectbox(
            "Filter by job",
            ["All"] + sorted(df["job"].dropna().unique().tolist())
        )

    with right:
        selected_education = st.selectbox(
            "Filter by education",
            ["All"] + sorted(
                df["education"].dropna().unique().tolist()
            )
        )

    filtered = df.copy()

    if selected_job != "All":
        filtered = filtered[filtered["job"] == selected_job]

    if selected_education != "All":
        filtered = filtered[
            filtered["education"] == selected_education
        ]

    if filtered.empty:
        st.warning("No customers match these filters.")
    else:
        n_customers = len(filtered)
        n_subscribers = int((filtered["y"] == "yes").sum())
        rate = n_subscribers / n_customers * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers in segment", f"{n_customers:,}")
        c2.metric("Historical subscribers", f"{n_subscribers:,}")
        c3.metric("Historical response rate", f"{rate:.2f}%")

        response_by_job = (
            filtered.groupby("job")["y"]
            .apply(lambda values: (values == "yes").mean() * 100)
            .reset_index(name="Response rate (%)")
            .sort_values("Response rate (%)", ascending=False)
        )

        fig = px.bar(
            response_by_job,
            x="job",
            y="Response rate (%)",
            title="Historical response rate by job",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)

        balance_plot = px.histogram(
            filtered,
            x="balance",
            color="y",
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            title="Account balance distribution by outcome"
        )
        st.plotly_chart(balance_plot, use_container_width=True)

        st.caption(
            "Segment response rates are historical associations, not "
            "causal effects of customer characteristics."
        )


# =========================================================
# 9. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        "Held-out test results from the current LightGBM model. "
        "These values are recorded from the model evaluation notebook."
    )

    metrics = {
        "Accuracy": 0.8708,
        "Precision": 0.4563,
        "Recall": 0.5435,
        "F1-score": 0.4961,
        "ROC-AUC": 0.8053
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Test F1-score", f"{metrics['F1-score']:.4f}")
    c2.metric("Test ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
    c3.metric("Decision threshold", f"{threshold:.2f}")

    metric_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score": list(metrics.values())
    })

    fig = px.bar(
        metric_df,
        x="Metric",
        y="Score",
        range_y=[0, 1],
        text_auto=".4f",
        title="Held-out test metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("How to interpret these results")

    st.markdown(
        """
        - **Precision (45.63%):** Of customers predicted as subscribers,
          about 46% were actual subscribers in the test set.
        - **Recall (54.35%):** The model identified about 54% of actual
          subscribers in the test set.
        - **F1-score (0.4961):** The balance of precision and recall at
          the evaluated threshold.
        - **ROC-AUC (0.8053):** Measures ranking discrimination across
          thresholds; it is not the accuracy of an individual probability.
        """
    )

    st.warning(
        "These are the recorded evaluation metrics for the current model "
        "and threshold. If the model is retrained or the evaluation split "
        "changes, update these values. This dashboard does not recalculate "
        "test metrics automatically."
    )

    st.subheader("Project limitations")

    st.markdown(
        """
        1. The model predicts historical subscription response, not causal
           campaign lift.
        2. It does not calculate ROI because reliable campaign cost,
           revenue, and margin data are not included.
        3. Predictions depend on the data distribution and quality.
        4. Historical performance does not guarantee future performance.
        """
    )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()
st.caption(
    "MarketingIQ | Educational decision-support MVP | "
    "Predictions are estimates, not guarantees."
)
