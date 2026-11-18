import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Academic Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

folder = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(
        folder / "outputs/student_risk_model.pkl"
    )

model = load_model()

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #1e293b;
}
[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}
h1, h2, h3 {
    color: #60a5fa !important;
}
.stButton > button,
.stFormSubmitButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎓 About the Project")
st.sidebar.write("**Developer:** MD SK MUSTAKIM")
st.sidebar.write("**Department:** CSE (AI & ML)")
st.sidebar.write("**Institute:** GKCIET, Malda")
st.sidebar.write("**Subject:** Introduction To Machine Learning")
st.sidebar.write("**Subject Code:** PCCAIML 502")
st.sidebar.write("**Guided by:** Dr. Showmik Bhowmik")
st.sidebar.divider()
st.sidebar.write("**Selected Model:** Logistic Regression")
st.sidebar.markdown(
    "[GitHub Repository](https://github.com/mdskmustakim/Student_Performance_Project)"
)

# Header
st.title("🎓 Student Academic Risk Predictor")
st.subheader("Early Warning System for Student Academic Performance")

st.write(
    "An intelligent Machine Learning prototype "
    "for identifying students who may need academic support."
)

# Dashboard
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", "649")
c2.metric("ML Algorithms", "3")
c3.metric("Accuracy", "76.92%")
c4.metric("Recall", "60%")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔍 Risk Prediction",
    "📊 Model Performance",
    "ℹ️ About Project"
])

# Friendly feature names
labels = {
    "school": "School",
    "sex": "Gender",
    "age": "Student Age",
    "address": "Residential Area",
    "famsize": "Family Size",
    "Pstatus": "Parents' Living Status",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "Mjob": "Mother's Occupation",
    "Fjob": "Father's Occupation",
    "reason": "Reason for School Selection",
    "guardian": "Guardian",
    "traveltime": "School Travel Time",
    "studytime": "Weekly Study Time",
    "failures": "Previous Academic Failures",
    "schoolsup": "Extra Educational Support",
    "famsup": "Family Support",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery School",
    "higher": "Wants Higher Education",
    "internet": "Internet Access",
    "romantic": "Romantic Relationship",
    "famrel": "Family Relationship",
    "freetime": "Free Time",
    "goout": "Going Out with Friends",
    "Dalc": "Weekday Alcohol Consumption",
    "Walc": "Weekend Alcohol Consumption",
    "health": "Current Health Condition",
    "absences": "School Absences",
    "G1": "First Period Grade",
    "G2": "Second Period Grade"
}

# Meaningful numerical options
education = {
    0: "None",
    1: "Primary Education",
    2: "5th to 9th Grade",
    3: "Secondary Education",
    4: "Higher Education"
}

choices = {
    "Medu": education,
    "Fedu": education,
    "studytime": {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours"
    },
    "traveltime": {
        1: "Less than 15 minutes",
        2: "15 to 30 minutes",
        3: "30 to 60 minutes",
        4: "More than 60 minutes"
    }
}

for field in ["famrel", "freetime", "goout",
              "Dalc", "Walc", "health"]:
    choices[field] = {
        1: "1 - Very Low",
        2: "2 - Low",
        3: "3 - Medium",
        4: "4 - High",
        5: "5 - Very High"
    }

category_labels = {
    "F": "Female",
    "M": "Male",
    "U": "Urban",
    "R": "Rural",
    "T": "Living Together",
    "A": "Living Apart",
    "LE3": "3 or Fewer Members",
    "GT3": "More than 3 Members",
    "yes": "Yes",
    "no": "No",
    "GP": "Gabriel Pereira",
    "MS": "Mousinho da Silveira"
}

# Risk prediction tab
with tab1:

    st.header("📝 Student Information")
    st.caption("Enter the student details below.")

    student = {}

    with st.form("student_form"):

        left, right = st.columns(2)
        i = 0

        for name, transformer, columns in model.named_steps[
            "preprocess"
        ].transformers_:

            if name == "num":

                defaults = transformer.named_steps[
                    "fill"
                ].statistics_

                for col, default in zip(columns, defaults):

                    panel = left if i % 2 == 0 else right

                    with panel:

                        label = labels.get(col, col)

                        if col in choices:
                            options = list(choices[col])

                            student[col] = st.selectbox(
                                label,
                                options,
                                index=options.index(
                                    int(default)
                                ) if int(default) in options else 0,
                                format_func=lambda v, c=col:
                                    choices[c][v],
                                key=col
                            )

                        else:

                            limits = {
                                "age": (15, 22),
                                "failures": (0, 3),
                                "absences": (0, 93),
                                "G1": (0, 20),
                                "G2": (0, 20)
                            }

                            minimum, maximum = limits.get(
                                col, (0, 100)
                            )

                            student[col] = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=max(
                                    minimum,
                                    min(maximum, int(default))
                                ),
                                step=1,
                                key=col
                            )

                    i += 1

            elif name == "cat":

                categories = transformer.named_steps[
                    "encode"
                ].categories_

                defaults = transformer.named_steps[
                    "fill"
                ].statistics_

                for col, options, default in zip(
                    columns, categories, defaults
                ):

                    panel = left if i % 2 == 0 else right
                    options = list(options)

                    with panel:

                        student[col] = st.selectbox(
                            labels.get(col, col),
                            options,
                            index=options.index(default)
                            if default in options else 0,
                            format_func=lambda v:
                                category_labels.get(
                                    v, str(v).replace("_", " ").title()
                                ),
                            key=col
                        )

                    i += 1

        submitted = st.form_submit_button(
            "🔍 Predict Academic Risk",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([student])
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)[0]
        score = model.predict_proba(input_data)[0][1]

        st.session_state["result"] = (
            prediction, score
        )

    if "result" in st.session_state:

        prediction, score = st.session_state["result"]

        st.divider()
        st.header("📊 Prediction Result")

        if prediction == 1:

            st.error("⚠️ Student is AT RISK")

            st.write("### Recommended Support")
            st.write("- Additional academic guidance")
            st.write("- Regular progress monitoring")
            st.write("- Improved study habits")
            st.write("- Attendance monitoring")

        else:

            st.success("✅ Student is NOT AT RISK")
            st.write(
                "Continue regular academic monitoring "
                "and maintain healthy study habits."
            )

        st.metric(
            "Model Risk Score",
            f"{score * 100:.2f}%"
        )

        st.progress(float(score))

        st.caption(
            "This score is not a calibrated real-world "
            "probability. For educational demonstration only."
        )

# Model performance tab
with tab2:

    st.header("📈 Machine Learning Model Comparison")

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy (%)": [76.92, 78.46, 84.62],
        "Precision (%)": [35.29, 21.43, 50.00],
        "Recall (%)": [60.00, 15.00, 30.00],
        "F1 Score (%)": [44.44, 17.65, 37.50]
    })

    st.dataframe(results, hide_index=True)

    st.bar_chart(
        results.set_index("Model")[
            ["Accuracy (%)", "Recall (%)", "F1 Score (%)"]
        ]
    )

    st.subheader("Confusion Matrix")

    st.image(
        str(folder / "outputs/confusion_matrix.png"),
        caption="Logistic Regression Confusion Matrix"
    )

    st.info(
        "Logistic Regression was selected for its "
        "higher Recall and F1 Score among the tested models."
    )

# About project tab
with tab3:

    st.header("About This Project")

    st.markdown("""
    ### Problem Statement
    Identify students who may be at risk of poor
    academic performance.

    ### Dataset
    - UCI Student Performance Dataset
    - Total records: 649
    - At Risk: Final grade below 10
    - Not At Risk: Final grade 10 or above

    ### Methodology
    - Data Cleaning
    - Data Preprocessing
    - Train-Test Split (80:20)
    - Model Training and Comparison
    - Academic Risk Prediction

    ### Algorithms
    - Logistic Regression
    - Decision Tree
    - Random Forest

    ### Limitations
    - The selected model has 60% Recall on the test set.
    - Some at-risk students may not be detected.
    - The dataset is from Portuguese schools, not GKCIET.
    - Independent validation is needed before real-world use.
    - Predictions should not replace teachers' judgement.
    """)

    st.markdown(
        "[Dataset Source – UCI Repository]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

st.divider()
st.caption(
    "Developed by MD SK MUSTAKIM | "
    "CSE (AI & ML) | GKCIET | 2026"
)

# Dataset Analysis and Model Performance

st.markdown("---")
st.header("📊 Dataset Analysis & Model Performance")

with st.expander("📚 Exploratory Data Analysis"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Risk Distribution")
        st.image("outputs/risk_distribution.png")

    with col2:
        st.subheader("Final Grade Distribution")
        st.image("outputs/grade_distribution.png")

with st.expander("📈 ML Model Performance Comparison"):

    st.image("outputs/model_comparison.png")

    results = pd.read_csv("outputs/model_results.csv")
    st.dataframe(results.round(4))

    st.info(
        "Logistic Regression is selected because "
        "it achieved the highest Recall and F1 Score "
        "among the three evaluated models."
    )

with st.expander("🎯 Confusion Matrix"):

    st.image("outputs/confusion_matrix.png")