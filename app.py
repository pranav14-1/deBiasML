import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load all necessary files 
model = pickle.load(open('resume_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

with open("skills_list.pkl", "rb") as f:
    skills_list = pickle.load(f)

with open("locations_list.pkl", "rb") as f:
    locations_list = pickle.load(f)

with open("companies_list.pkl", "rb") as f:
    companies_list = pickle.load(f)

# Try to load training data for feedback
try:
    X_train = pickle.load(open("X_train.pkl", "rb"))
    Y_train = pickle.load(open("Y_train.pkl", "rb"))
    feedback_enabled = True
except:
    feedback_enabled = False

# Better Naming
feature_name_map = {
    'project_count': "Number of Projects",
    'internship_count': "Number of Internships",
    'has_python': "Python",
    'has_java': "Java",
    'has_ml': "Machine Learning",
    'has_dsa': "Data Structures & Algorithms",
    'has_html': "HTML",
    'has_css': "CSS",
    'has_javascript': "JavaScript",
    'has_sql': "SQL",
    'has_tableau': "Tableau",
    'has_power bi': "Power BI"
}

# Auto-map for one-hot encoded columns
for loc in locations_list:
    feature_name_map[f'location_{loc}'] = f"Location: {loc}"

for comp in companies_list:
    feature_name_map[f'target_company_{comp}'] = f"Target Company: {comp}"

for et in ['Intern', 'Freelance', 'Full-Time']:
    feature_name_map[f'experience_type_{et}'] = f"Experience Type: {et}"

# Feedback Function 
def generate_feedback(user_vector, model, X_train, Y_train, top_n=5):
    shortlisted = X_train[Y_train == 1]
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    unchangeable_features = ['college_tier', 'gender']
    feedback = []

    for feat in feature_importances.index:
        if feat in unchangeable_features:
            continue

        user_val = user_vector.get(feat, 0)
        avg_val = shortlisted[feat].mean()

        if user_val < avg_val:
            friendly_name = feature_name_map.get(feat, feat.replace('_', ' ').capitalize())
            feedback.append(f"You may improve your chances by improving: {friendly_name}")

        if len(feedback) >= top_n:
            break

    return feedback


# Streamlit App Starts Here 
st.set_page_config(page_title="Resume Shortlisting Predictor", layout="centered")
st.title("📄 Resume Shortlisting Predictor")
st.markdown("Predict your chances of being shortlisted — and get personalized feedback to improve your resume!")

# User Inputs 
gender = st.radio("👤 Gender", ['Male', 'Female'])
college_tier = st.selectbox("🎓 College Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
experience_type = st.selectbox("🧪 Experience Type", ['Intern', 'Freelance', 'Full-Time'])
location = st.selectbox("📍 Location", locations_list)
target_company = st.selectbox("🏢 Target Company", companies_list)
selected_skills = st.multiselect("🛠️ Select Your Skills", skills_list)
project_count = st.number_input("📊 Number of Projects", min_value=0, max_value=20, step=1)
internship_count = st.number_input("💼 Number of Internships", min_value=0, max_value=10, step=1)

# Submit 
if st.button("🚀 Check My Resume Fit"):
    input_dict = {}

    # Label encoding
    input_dict['gender'] = 0 if gender == 'Male' else 1
    input_dict['college_tier'] = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}[college_tier]

    # One-hot encoding
    for loc in locations_list:
        input_dict[f'location_{loc}'] = 1 if location == loc else 0

    for et in ['Intern', 'Freelance', 'Full-Time']:
        input_dict[f'experience_type_{et}'] = 1 if experience_type == et else 0

    for comp in companies_list:
        input_dict[f'target_company_{comp}'] = 1 if target_company == comp else 0

    # Multi-hot skills
    for skill in skills_list:
        input_dict[f'has_{skill}'] = 1 if skill in selected_skills else 0

    # Numeric values
    input_dict['project_count'] = project_count
    input_dict['internship_count'] = internship_count

    # Fill missing columns with 0
    for col in columns:
        if col not in input_dict:
            input_dict[col] = 0

    # Create input DataFrame
    user_df = pd.DataFrame([input_dict])[columns]

    # Predict
    pred = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][1]
    percent = round(prob * 100, 2)

    # Show result
    st.metric(label="🎯 Selection Probability", value=f"{percent}%")
    if pred == 1:
        st.success("✅ You are likely to be shortlisted!")
    else:
        st.error("❌ You may not get shortlisted.")

    # Feedback
    st.divider()
    st.subheader("🧠 Resume Suggestions")
    if feedback_enabled:
        fb = generate_feedback(input_dict, model, X_train, Y_train)
        if fb:
            for tip in fb:
                st.write("🔸", tip)
        else:
            st.success("Your resume looks strong compared to shortlisted candidates!")
    else:
        st.info("⚠️ Feedback unavailable — training data not loaded.")
