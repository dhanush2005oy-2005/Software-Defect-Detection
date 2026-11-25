import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model
with open("software_defect_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("software_defect_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="💻",
    layout="wide"
)
if "reset" not in st.session_state:
    st.session_state.reset = False

# Custom CSS

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f5f7fb;
}

/* Main heading */
h1 {
    color: #17365D;
    font-weight: 700;
}

/* Section headings */
h2, h3 {
    color: #24476b;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8eef7;
}

/* Input fields */
.stNumberInput {
    background-color: white;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #28578a;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Main title
st.title("Software Defect Prediction System")

st.write(
    "Predict whether a software module contains defects "
    "using Machine Learning."
)

st.divider()

# Sidebar
st.sidebar.title("About Project")
st.sidebar.write(
    "This application uses Machine Learning "
    "to predict software defects."
)

# Main section
st.header("Enter Software Metrics")

st.info("Enter the software metrics below to predict defects.")

# Input fields

col1, col2, col3 = st.columns(3)

with col1:
    loc = st.number_input("Lines of Code (loc)", min_value=0.0)
    v_g = st.number_input("Cyclomatic Complexity (v(g))", min_value=0.0)
    ev_g = st.number_input("Essential Complexity (ev(g))", min_value=0.0)
    iv_g = st.number_input("Design Complexity (iv(g))", min_value=0.0)
    n = st.number_input("Halstead Length (n)", min_value=0.0)
    v = st.number_input("Halstead Volume (v)", min_value=0.0)
    l = st.number_input("Halstead Program Length (l)", min_value=0.0)

with col2:
    d = st.number_input("Halstead Difficulty (d)", min_value=0.0)
    i = st.number_input("Halstead Intelligence (i)", min_value=0.0)
    e = st.number_input("Halstead Effort (e)", min_value=0.0)
    b = st.number_input("Halstead Bugs (b)", min_value=0.0)
    t = st.number_input("Halstead Time (t)", min_value=0.0)
    lOCode = st.number_input("Lines of Code (lOCode)", min_value=0.0)
    lOComment = st.number_input("Lines of Comments (lOComment)", min_value=0.0)

with col3:
    lOBlank = st.number_input("Blank Lines (lOBlank)", min_value=0.0)
    locCodeAndComment = st.number_input("Code and Comment Lines", min_value=0.0)
    uniq_Op = st.number_input("Unique Operators (uniq_Op)", min_value=0.0)
    uniq_Opnd = st.number_input("Unique Operands (uniq_Opnd)", min_value=0.0)
    total_Op = st.number_input("Total Operators (total_Op)", min_value=0.0)
    total_Opnd = st.number_input("Total Operands (total_Opnd)", min_value=0.0)
    branchCount = st.number_input("Branch Count (branchCount)", min_value=0.0)

def reset_inputs():
    keys = [
        "loc", "v_g", "ev_g", "iv_g", "n", "v", "l",
        "d", "i", "e", "b", "t", "lOCode", "lOComment",
        "lOBlank", "locCodeAndComment", "uniq_Op",
        "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"
    ]

    for key in keys:
        st.session_state[key] = 0.0

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("Predict Software Defect", use_container_width=True)

with col2:
    st.button("Reset Inputs", on_click=reset_inputs, use_container_width=True)

if predict_button:

    input_data = pd.DataFrame([[
        loc, v_g, ev_g, iv_g, n, v, l,
        d, i, e, b, t, lOCode, lOComment,
        lOBlank, locCodeAndComment, uniq_Op,
        uniq_Opnd, total_Op, total_Opnd, branchCount
    ]], columns=[
        'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l',
        'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment',
        'lOBlank', 'locCodeAndComment', 'uniq_Op',
        'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'
    ])

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]

    predicted_class = prediction[0]
    class_index = list(model.classes_).index(predicted_class)
    probability = probabilities[class_index] * 100

    with st.container(border=True):

        st.subheader("Prediction Result")

        if predicted_class == 1:
            st.error("DEFECTIVE SOFTWARE")
        else:
            st.success("NON-DEFECTIVE SOFTWARE")

        st.metric(
            label="Model-estimated probability",
            value=f"{probability:.2f}%"
        )

        st.progress(float(probability) / 100)

        st.caption(
            "This probability is the model's estimated confidence, "
            "not a guarantee of actual software quality."
        )