"""
app.py
Streamlit web application — UCLA Admission Prediction (Neural Network / MLP).
Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    DATA_PATH, MODEL_PATH, SCALER_PATH, ADMIT_THRESHOLD,
    ACTIVATION_OPTIONS, DEFAULT_HIDDEN_LAYER_SIZES,
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_MAX_ITER,
)
from src.logger import get_logger
from src.data_loader import load_data
from src.preprocessing import (
    binarize_target, drop_unused_columns, encode_features, split_features_target,
)
from src.train import run_training_pipeline, load_artifacts
from src.predict import predict_admission

logger = get_logger(__name__)

st.set_page_config(
    page_title="UCLA Admission Predictor",
    page_icon="🎓",
    layout="wide",
)


# ── Cached data helpers ───────────────────────────────────────────────────────

@st.cache_data
def get_raw_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_processed_data():
    df = load_data(DATA_PATH)
    df = binarize_target(df)
    df = drop_unused_columns(df)
    df = encode_features(df)
    X, y = split_features_target(df)
    return X, y


# ── Sidebar nav ──────────────────────────────────────────────────────────────
st.sidebar.title("🎓 UCLA Admission")
page = st.sidebar.radio(
    "Navigate",
    ["Data Explorer", "Model Training", "Predict"],
)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Explorer
# ════════════════════════════════════════════════════════════════════════════
if page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown(
        "Explore the **UCLA Admission dataset** — 500 applicant profiles used "
        "to predict admission likelihood."
    )

    df_raw = get_raw_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Applicants", df_raw.shape[0])
    col2.metric("Features", df_raw.shape[1] - 2)  # exclude Serial_No & target
    admit_rate = (df_raw["Admit_Chance"] >= ADMIT_THRESHOLD).mean()
    col3.metric("Admit Rate", f"{admit_rate:.1%}", help=f"Admit_Chance ≥ {ADMIT_THRESHOLD}")

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df_raw.describe().round(3), use_container_width=True)

    st.subheader("Admit_Chance Distribution")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df_raw["Admit_Chance"], bins=25, color="#3498db", edgecolor="white")
    ax.axvline(ADMIT_THRESHOLD, color="red", linestyle="--",
              label=f"Admit threshold ({ADMIT_THRESHOLD})")
    ax.set_xlabel("Admit_Chance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Admission Chance")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("GRE vs TOEFL Score, coloured by Admission")
    df_plot = df_raw.copy()
    df_plot["Admit"] = (df_plot["Admit_Chance"] >= ADMIT_THRESHOLD).map({True: "Admit", False: "Reject"})
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_plot, x="GRE_Score", y="TOEFL_Score", hue="Admit",
        palette={"Admit": "#2ecc71", "Reject": "#e74c3c"}, ax=ax2, alpha=0.75,
    )
    ax2.set_title("GRE vs TOEFL — Admitted vs Rejected")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("Feature Distributions")
    num_cols = ["GRE_Score", "TOEFL_Score", "CGPA", "SOP", "LOR"]
    fig3, axes = plt.subplots(1, len(num_cols), figsize=(15, 3))
    for ax, col in zip(axes, num_cols):
        df_raw[col].hist(ax=ax, bins=20, color="#9b59b6", edgecolor="white")
        ax.set_title(col, fontsize=9)
    fig3.tight_layout()
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    corr = df_raw.drop(columns=["Serial_No"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4,
               linewidths=0.5, vmin=-1, vmax=1)
    fig4.tight_layout()
    st.pyplot(fig4)

    st.subheader("Categorical Feature Counts")
    fig5, axes5 = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, col in zip(axes5, ["University_Rating", "Research"]):
        vc = df_raw[col].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color="#1abc9c")
        ax.set_title(col, fontsize=9)
    fig5.tight_layout()
    st.pyplot(fig5)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Training
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.title("🧠 Neural Network Training & Evaluation")
    st.markdown(
        "Train a feed-forward **MLPClassifier** to predict admission. "
        "Tune the architecture in the sidebar, then train."
    )

    X, y = get_processed_data()

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Network Architecture")
    n_layers = st.sidebar.slider("Number of hidden layers", 1, 3, 1)
    layer_sizes = []
    for i in range(n_layers):
        size = st.sidebar.slider(f"Neurons — layer {i+1}", 2, 50,
                                 DEFAULT_HIDDEN_LAYER_SIZES[0] if i == 0 else 10)
        layer_sizes.append(size)
    hidden_layer_sizes = tuple(layer_sizes)

    activation = st.sidebar.selectbox("Activation function", ACTIVATION_OPTIONS,
                                      index=ACTIVATION_OPTIONS.index(DEFAULT_ACTIVATION))
    batch_size = st.sidebar.slider("Batch size", 8, 200, DEFAULT_BATCH_SIZE, step=8)
    max_iter   = st.sidebar.slider("Max iterations (epochs)", 50, 1000, DEFAULT_MAX_ITER, step=50)

    artifacts_exist = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
    if artifacts_exist and "results" not in st.session_state:
        st.info(
            "Pre-trained model found on disk. Click **Train** to (re)build with the chosen architecture.",
            icon="ℹ️",
        )

    train_btn = st.button("🚀 Train Model", type="primary")

    if train_btn:
        with st.spinner(f"Training MLP {hidden_layer_sizes} with {activation} activation …"):
            results = run_training_pipeline(
                X, y,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                batch_size=batch_size,
                max_iter=max_iter,
            )
        st.session_state["results"] = results
        st.success("Training complete!", icon="✅")

    if "results" in st.session_state:
        results = st.session_state["results"]

        st.subheader("Architecture Used")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hidden Layers", str(results["hidden_layer_sizes"]))
        c2.metric("Activation", results["activation"])
        c3.metric("Batch Size", results["batch_size"])
        c4.metric("Max Iterations", results["max_iter"])

        st.subheader("Accuracy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Accuracy", f"{results['train_eval']['accuracy']:.4f}")
        c2.metric("Test Accuracy",  f"{results['test_eval']['accuracy']:.4f}")
        c3.metric("CV Mean (±std)",
                  f"{results['cv']['mean']:.4f} (±{results['cv']['std']:.4f})")

        target_met = results["test_eval"]["accuracy"] >= 0.90
        if target_met:
            st.success("Success criteria met: test accuracy ≥ 90%", icon="🎯")
        else:
            st.warning(
                "Test accuracy is below the 90% success criteria. "
                "Try adjusting the architecture, activation, or iterations.",
                icon="⚠️",
            )

        # ── Loss curve ───────────────────────────────────────────────────
        st.subheader("Training Loss Curve")
        fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
        ax_loss.plot(results["loss_curve"], color="#2980b9", linewidth=1.5)
        ax_loss.set_xlabel("Iterations")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("MLP Training Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        fig_loss.tight_layout()
        st.pyplot(fig_loss)

        # ── Confusion matrices ───────────────────────────────────────────
        st.subheader("Confusion Matrices")
        tab_train, tab_test = st.tabs(["Train Set", "Test Set"])

        for tab, key, name in [(tab_train, "train_eval", "Train"),
                                (tab_test, "test_eval", "Test")]:
            with tab:
                cm = results[key]["confusion_matrix"]
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Reject", "Admit"],
                    yticklabels=["Reject", "Admit"],
                    ax=ax_cm,
                )
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title(f"{name} Set — Confusion Matrix")
                fig_cm.tight_layout()
                st.pyplot(fig_cm)

                st.write("**Classification Report:**")
                report_df = pd.DataFrame(results[key]["report"]).T
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Cross-Validation Fold Scores")
        st.write(", ".join(f"{s:.4f}" for s in results["cv"]["scores"]))

    else:
        st.info("Set the architecture in the sidebar and click **Train Model** to see results.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("🔮 Predict Admission Chance")
    st.markdown(
        "Enter an applicant's profile below to predict whether they would "
        "be admitted to UCLA."
    )

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.warning(
            "No trained model found. Go to **🧠 Model Training** and train the model first.",
            icon="⚠️",
        )
        st.stop()

    # Recompute feature columns (cheap, cached)
    X, y = get_processed_data()
    feature_cols = list(X.columns)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gre_score    = st.slider("GRE Score (out of 340)",   260, 340, 316)
            toefl_score  = st.slider("TOEFL Score (out of 120)", 90,  120, 107)
            cgpa         = st.slider("CGPA (out of 10)",         6.0, 10.0, 8.6, step=0.01)
            research     = st.selectbox("Research Experience", [1, 0],
                                        format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            university_rating = st.selectbox("University Rating (1-5)", [1, 2, 3, 4, 5], index=2)
            sop = st.slider("Statement of Purpose Strength (out of 5)", 1.0, 5.0, 3.5, step=0.5)
            lor = st.slider("Letter of Recommendation Strength (out of 5)", 1.0, 5.0, 3.5, step=0.5)

        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        raw_input = {
            "GRE_Score":          gre_score,
            "TOEFL_Score":        toefl_score,
            "University_Rating":  university_rating,
            "SOP":                sop,
            "LOR":                lor,
            "CGPA":               cgpa,
            "Research":           research,
        }

        try:
            result = predict_admission(raw_input, feature_cols)
            label  = result["label"]
            prob   = result["probability"]

            if label == "Admit":
                st.success(f"**{label}** — Congratulations!!!! Probability: {prob:.1%}", icon="✅")
            else:
                st.error(f"**{label}** — Opps!!!! Probability: {prob:.1%}", icon="❌")

            fig_g, ax_g = plt.subplots(figsize=(5, 0.6))
            ax_g.barh(["P(Admit)"], [prob], color="#2ecc71" if label == "Admit" else "#e74c3c")
            ax_g.barh(["P(Admit)"], [1 - prob], left=[prob], color="#ecf0f1")
            ax_g.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
            ax_g.set_xlim(0, 1)
            ax_g.set_xlabel("Probability")
            fig_g.tight_layout()
            st.pyplot(fig_g)


        except Exception as exc:
            st.error(f"Prediction failed: {exc}", icon="⚠️")
            logger.exception("Prediction error")
