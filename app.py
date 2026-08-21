from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Telecom Churn Model1.csv"
MODEL_PATH = ROOT / "artifacts" / "telecom_churn_model.joblib"
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📱", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

def risk_band(probability):
    return "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"

def recommended_actions(customer, probability):
    actions = []
    if customer.get("Contract") == "Month-to-month":
        actions.append("Ask whether a longer-term plan with a suitable incentive would be helpful.")
    if customer.get("TechSupport") == "No":
        actions.append("Check whether the customer needs help setting up or using technical support.")
    if customer.get("OnlineSecurity") == "No":
        actions.append("Explain the available online-security support in simple terms.")
    if float(customer.get("tenure", 0)) < 12:
        actions.append("Arrange a friendly check-in because the customer is still relatively new.")
    if probability >= 0.40:
        actions.append("Ask about service concerns and whether the current plan still suits the customer's budget.")
    return actions or ["Keep the customer engaged and watch for any change in service needs."]

try:
    df, bundle = load_data(), load_bundle()
except FileNotFoundError as exc:
    st.error(f"Required project file is missing: {exc.filename}")
    st.stop()

model = bundle["model"]
threshold = float(bundle["threshold"])
metrics = bundle["metrics"]
feature_columns = bundle["feature_columns"]

st.title("📱 Telecom Churn Prediction")
st.caption("3MTT Capstone Project · Sunday Babatunde · FE/26/2850020886 · 4peace1@gmail.com · Male")
st.info("This app was built for learning. The score helps to identify customers worth checking in with; it cannot tell us for certain that someone will leave or explain the reason.")
overview_tab, prediction_tab, model_tab = st.tabs(["Customer overview", "Predict churn", "Model performance"])

with overview_tab:
    churn_numeric = df["Churn"].map({"Yes": 1, "No": 0})
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Observed churners", f"{int(churn_numeric.sum()):,}")
    c3.metric("Observed churn rate", f"{churn_numeric.mean():.1%}")
    contract_rates = (df.assign(ChurnFlag=churn_numeric)
                      .groupby("Contract", observed=True)["ChurnFlag"].mean()
                      .sort_values(ascending=False))
    st.subheader("Observed churn rate by contract")
    st.bar_chart(contract_rates, y_label="Churn rate")
    st.write("I used the public IBM-style Telco Customer Churn sample for this project. It is useful for demonstrating the process, but it should not be treated as data from a Nigerian network provider.")

with prediction_tab:
    st.subheader("Enter a customer profile")
    input_df = df.drop(columns=["customerID", "Churn"])
    customer = {}
    with st.form("prediction_form"):
        columns = st.columns(3)
        for index, column in enumerate(feature_columns):
            series = input_df[column]
            with columns[index % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    if column == "SeniorCitizen":
                        customer[column] = st.selectbox(column, [0, 1], format_func=lambda x: "Yes" if x else "No")
                    else:
                        customer[column] = st.number_input(column, min_value=float(series.min()),
                            max_value=float(series.max()), value=float(series.median()))
                else:
                    customer[column] = st.selectbox(column, sorted(series.dropna().astype(str).unique()))
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    if submitted:
        customer_frame = pd.DataFrame([customer], columns=feature_columns)
        probability = float(model.predict_proba(customer_frame)[0, 1])
        predicted_churn = probability >= threshold
        left, right = st.columns([1, 2])
        left.metric("Churn probability", f"{probability:.1%}")
        left.metric("Risk band", risk_band(probability))
        if predicted_churn:
            right.warning(f"This customer may be worth a follow-up conversation (review threshold: {threshold:.0%}).")
        else:
            right.success(f"The score is below the current review threshold of {threshold:.0%}.")
        right.markdown("**Suggested next actions**")
        for action in recommended_actions(customer, probability):
            right.write(f"- {action}")

with model_tab:
    st.subheader("Validated holdout performance")
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
        ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{metrics[key]:.1%}")
    st.caption(f"I selected the {threshold:.0%} threshold on the validation data because I wanted to find more actual churners. These results come from the separate 20% test set.")
    cm = bundle["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stayed", "Churned"],
           yticklabels=["Stayed", "Churned"], xlabel="Predicted", ylabel="Actual",
           title="Confusion matrix on holdout data")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)
    st.warning("The lower threshold catches more churners, but it also flags more people who would have stayed. A real retention team would need to balance this against its budget and contact capacity.")

st.divider()
st.caption("© 2026 Sunday Babatunde · Educational use only")
