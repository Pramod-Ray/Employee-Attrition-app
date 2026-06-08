# =========================================================
# EMPLOYEE ATTRITION PREDICTION SYSTEM
# Built with XGBoost + SMOTE + Streamlit
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 40px 35px;
    border-radius: 16px;
    margin-bottom: 28px;
    border-left: 5px solid #e94560;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 { color: #ffffff; font-size: 2.2rem; margin: 0 0 10px 0; font-weight: 700; }
.hero p  { color: #a8b2d8; font-size: 1.05rem; margin: 0; }

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
    padding: 20px 24px;
    border-radius: 12px;
    border-top: 4px solid #e94560;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 16px;
}
.metric-card .label { font-size: 0.85rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-top: 4px; }

.section-header {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    color: white;
    padding: 14px 22px;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 20px 0 16px 0;
    border-left: 4px solid #e94560;
}

.result-leave {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white; padding: 22px; border-radius: 14px;
    text-align: center; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(255,65,108,0.4);
    margin-top: 16px;
}
.result-stay {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white; padding: 22px; border-radius: 14px;
    text-align: center; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(17,153,142,0.4);
    margin-top: 16px;
}

.reason-tag {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 3px;
}
.reason-good {
    background: #d4edda;
    color: #155724;
    border: 1px solid #28a745;
}

.benefit-card {
    background: linear-gradient(135deg, #f0f4ff, #e8f0fe);
    border: 1px solid #c7d7fc;
    border-left: 5px solid #3b5bdb;
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.benefit-card .benefit-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.benefit-card .benefit-desc {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.5;
}

.sidebar-card {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: white;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
    border-left: 3px solid #e94560;
}

.stButton > button {
    background: linear-gradient(135deg, #e94560, #c62a47);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    box-shadow: 0 4px 15px rgba(233,69,96,0.35);
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD DATA ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Employee-Attrition.csv")
    return df


# ─── TRAIN MODEL ─────────────────────────────────────────
@st.cache_resource
def train_model():
    df = load_data()
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])

    features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
        'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]

    X = df[features]
    y = df['Attrition']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1, reg_lambda=1,
        random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, features, accuracy


# ─── REASONS LOGIC ───────────────────────────────────────
def get_reasons(data: dict):
    reasons = []
    if data.get("MonthlyIncome", 9999) < 3000:
        reasons.append(("⚠️ Low Salary", "warning"))
    if data.get("JobSatisfaction", 5) <= 2:
        reasons.append(("😞 Low Job Satisfaction", "warning"))
    if data.get("EnvironmentSatisfaction", 5) <= 2:
        reasons.append(("🏚️ Poor Work Environment", "warning"))
    if data.get("WorkLifeBalance", 5) <= 2:
        reasons.append(("⏰ Poor Work-Life Balance", "warning"))
    if data.get("DistanceFromHome", 0) > 20:
        reasons.append(("🚗 Long Commute Distance", "warning"))
    if data.get("NumCompaniesWorked", 0) >= 5:
        reasons.append(("🔄 Frequently Switched Companies", "warning"))
    if data.get("YearsSinceLastPromotion", 0) >= 5:
        reasons.append(("📉 No Recent Promotion", "warning"))
    if not reasons:
        reasons.append(("✅ Employee Profile Looks Stable", "good"))
    return reasons


# ─── COMPANY BENEFITS LOGIC ──────────────────────────────
def get_company_benefits(data: dict) -> list:
    benefits = []

    if data.get("MonthlyIncome", 9999) < 3000:
        benefits.append((
            "💰 Competitive Salary & Bonus Structure",
            "Revise the employee's CTC based on market benchmarks. "
            "Introduce performance-linked quarterly bonuses and annual increment policies "
            "to ensure financial satisfaction and reduce turnover."
        ))

    if data.get("JobSatisfaction", 5) <= 2:
        benefits.append((
            "🎯 Role Enrichment & Career Clarity",
            "Assign more meaningful, challenging tasks aligned with the employee's strengths. "
            "Define a clear job roadmap with KPIs and provide regular constructive feedback "
            "through monthly 1:1 manager meetings."
        ))

    if data.get("EnvironmentSatisfaction", 5) <= 2:
        benefits.append((
            "🏢 Positive Workplace Culture Program",
            "Launch team-building activities, open-door leadership policies, and anonymous "
            "feedback channels. Recognize achievements publicly through 'Employee of the Month' "
            "programs to foster a sense of belonging."
        ))

    if data.get("WorkLifeBalance", 5) <= 2:
        benefits.append((
            "🕐 Flexible Work & Wellness Benefits",
            "Offer hybrid/remote work options, flexible shift timings, and mandatory paid leaves. "
            "Introduce mental health support programs, meditation sessions, and gym reimbursements "
            "to improve overall well-being."
        ))

    if data.get("DistanceFromHome", 0) > 20:
        benefits.append((
            "🚌 Commute Assistance & Relocation Support",
            "Provide company-sponsored cab/transport facilities or monthly travel allowance. "
            "Offer relocation assistance or housing allowance for employees living far from the office "
            "to reduce daily commute stress."
        ))

    if data.get("NumCompaniesWorked", 0) >= 5:
        benefits.append((
            "🤝 Employee Loyalty & Retention Rewards",
            "Introduce long-term retention bonuses (1-year, 3-year, 5-year milestones). "
            "Offer ESOPs (Employee Stock Ownership Plans) or loyalty increments to encourage "
            "commitment and reduce job-hopping behavior."
        ))

    if data.get("YearsSinceLastPromotion", 0) >= 5:
        benefits.append((
            "📈 Structured Promotion & Growth Path",
            "Create a transparent promotion policy with clear eligibility criteria. "
            "Offer internal job postings, leadership training programs, and fast-track promotion "
            "schemes for high-performing employees who have been stagnant in their roles."
        ))

    if data.get("TrainingTimesLastYear", 10) < 2:
        benefits.append((
            "📚 Learning & Development Programs",
            "Sponsor online courses, certifications, and industry conferences. "
            "Set up an internal Learning Management System (LMS) with curated skill-building content "
            "to help employees grow professionally within the organization."
        ))

    if data.get("RelationshipSatisfaction", 5) <= 2:
        benefits.append((
            "👥 Team Bonding & Peer Mentorship",
            "Assign a senior mentor to guide the employee. Organize cross-functional team projects, "
            "offsite retreats, and informal lunch/coffee meets to strengthen inter-personal relationships "
            "and team cohesion."
        ))

    if data.get("StockOptionLevel", 5) == 0:
        benefits.append((
            "📊 Stock Options & Financial Investment Benefits",
            "Offer Employee Stock Option Plans (ESOPs) or profit-sharing schemes. "
            "Provide provident fund matching, NPS contributions, and health insurance coverage "
            "to enhance the overall financial security of the employee."
        ))

    # Default benefit always shown
    benefits.append((
        "🌟 Employee Recognition & Engagement Program",
        "Launch a year-round recognition platform where peers and managers can appreciate "
        "contributions. Conduct quarterly engagement surveys, act on feedback, and celebrate "
        "milestones like work anniversaries and project completions to keep morale high."
    ))

    return benefits


# ─── LOAD MODEL ──────────────────────────────────────────
with st.spinner("Training model, please wait…"):
    model, scaler, features, accuracy = train_model()

# ─── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 Model Information")
    st.markdown(f"""
    <div class="sidebar-card">
        <div style="font-size:0.8rem;color:#a8b2d8;text-transform:uppercase;letter-spacing:1px;">Model Accuracy</div>
        <div style="font-size:2rem;font-weight:700;color:#fff;">{accuracy:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    

# ─── HERO ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📊 Employee Attrition Prediction System</h1>
    <p>Predict whether an employee may leave the company using Machine Learning & AI.</p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Bulk Prediction"])


# ══════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-header">👤 Employee Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        employee_name            = st.text_input("Employee Name", placeholder="e.g. Pramod Ray")
        employee_id              = st.text_input("Employee ID",   placeholder="e.g. EMP-001")
        department               = st.selectbox("Department", ["HR","IT","Sales","Finance","Marketing","Operations"])
        Age                      = st.number_input("Age", min_value=18, max_value=65, value=30)
        DistanceFromHome         = st.number_input("Distance From Home (km)", min_value=0, max_value=100, value=5)
        Education                = st.slider("Education", 1, 5, 3)
        EnvironmentSatisfaction  = st.slider("Environment Satisfaction", 1, 4, 3)
        JobSatisfaction          = st.slider("Job Satisfaction", 1, 4, 3)
        WorkLifeBalance          = st.slider("Work Life Balance", 1, 4, 3)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)

    with col2:
        DailyRate               = st.number_input("Daily Rate",     min_value=0, value=800)
        HourlyRate              = st.number_input("Hourly Rate",    min_value=0, value=60)
        MonthlyIncome           = st.number_input("Monthly Income", min_value=0, value=50000)
        MonthlyRate             = st.number_input("Monthly Rate",   min_value=0, value=14000)
        NumCompaniesWorked      = st.number_input("Number of Companies Worked",  min_value=0, max_value=20, value=1)
        PercentSalaryHike       = st.number_input("Percent Salary Hike",         min_value=0, max_value=100, value=15)
        PerformanceRating       = st.slider("Performance Rating", 1, 4, 3)
        StockOptionLevel        = st.number_input("Stock Option Level",          min_value=0, max_value=3, value=0)
        TotalWorkingYears       = st.number_input("Total Working Years",         min_value=0, max_value=40, value=5)
        TrainingTimesLastYear   = st.number_input("Training Times Last Year",    min_value=0, max_value=10, value=3)
        YearsAtCompany          = st.number_input("Years At Company",            min_value=0, max_value=40, value=3)
        YearsInCurrentRole      = st.number_input("Years In Current Role",       min_value=0, max_value=20, value=2)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion",  min_value=0, max_value=15, value=1)
        YearsWithCurrManager    = st.number_input("Years With Current Manager",  min_value=0, max_value=20, value=2)
        JobInvolvement          = st.slider("Job Involvement", 1, 4, 3)
        JobLevel                = st.slider("Job Level", 1, 5, 2)

    if st.button("🔍 Predict Employee Attrition"):

        input_values = {
            'Age': Age, 'DailyRate': DailyRate, 'DistanceFromHome': DistanceFromHome,
            'Education': Education, 'EnvironmentSatisfaction': EnvironmentSatisfaction,
            'HourlyRate': HourlyRate, 'JobInvolvement': JobInvolvement, 'JobLevel': JobLevel,
            'JobSatisfaction': JobSatisfaction, 'MonthlyIncome': MonthlyIncome,
            'MonthlyRate': MonthlyRate, 'NumCompaniesWorked': NumCompaniesWorked,
            'PercentSalaryHike': PercentSalaryHike, 'PerformanceRating': PerformanceRating,
            'RelationshipSatisfaction': RelationshipSatisfaction, 'StockOptionLevel': StockOptionLevel,
            'TotalWorkingYears': TotalWorkingYears, 'TrainingTimesLastYear': TrainingTimesLastYear,
            'WorkLifeBalance': WorkLifeBalance, 'YearsAtCompany': YearsAtCompany,
            'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion,
            'YearsWithCurrManager': YearsWithCurrManager
        }

        input_df     = pd.DataFrame([input_values])[features]
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        stay_prob    = proba[0] * 100
        leave_prob   = proba[1] * 100

        st.markdown("---")
        st.markdown('<div class="section-header">📈 Prediction Result</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Stay Probability</div>
                <div class="value" style="color:#11998e;">{stay_prob:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Leave Probability</div>
                <div class="value" style="color:#e94560;">{leave_prob:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(0, stay_prob,  color="#11998e", height=0.5)
        ax.barh(0, leave_prob, left=stay_prob,  color="#e94560", height=0.5)
        ax.set_xlim(0, 100)
        ax.axis("off")
        ax.text(stay_prob/2, 0, f"Stay {stay_prob:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        ax.text(stay_prob + leave_prob/2, 0, f"Leave {leave_prob:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if prediction == 1:
            st.markdown(
                f'<div class="result-leave">❌ {employee_name or "Employee"} is likely to LEAVE</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-stay">✅ {employee_name or "Employee"} is likely to STAY</div>',
                unsafe_allow_html=True
            )

        # ── Possible Reasons ──────────────────────────────
        st.markdown('<div class="section-header">🔎 Possible Reasons</div>', unsafe_allow_html=True)
        reasons = get_reasons(input_values)
        tags_html = "".join([
            f'<span class="reason-tag {"reason-good" if kind=="good" else ""}">{label}</span>'
            for label, kind in reasons
        ])
        st.markdown(tags_html, unsafe_allow_html=True)

        # ── Show only if WILL LEAVE ───────────────────────
        if prediction == 1:

            # HR Recommendations
            st.markdown('<div class="section-header">💡 HR Recommendations</div>', unsafe_allow_html=True)
            recs = []
            if MonthlyIncome < 3000:          recs.append("💰 Consider a salary revision or performance bonus.")
            if JobSatisfaction <= 2:           recs.append("🗣️ Schedule 1:1 feedback sessions and improve role clarity.")
            if EnvironmentSatisfaction <= 2:   recs.append("🏢 Address workplace culture and team dynamics.")
            if WorkLifeBalance <= 2:           recs.append("🕐 Offer flexible working hours or remote options.")
            if YearsSinceLastPromotion >= 5:   recs.append("📈 Review career progression and create a promotion roadmap.")
            if not recs:                        recs.append("🤝 Conduct a detailed stay interview to understand hidden concerns.")
            for r in recs:
                st.markdown(f"> {r}")

            # Company Benefits
            st.markdown('<div class="section-header">🏢 Company Benefits to Retain This Employee</div>', unsafe_allow_html=True)
            benefits = get_company_benefits(input_values)
            for title, desc in benefits:
                st.markdown(f"""
                <div class="benefit-card">
                    <div class="benefit-title">{title}</div>
                    <div class="benefit-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — BULK PREDICTION
# ══════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-header">📂 Bulk Employee Prediction</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Employee CSV File", type=["csv"])

    if uploaded_file is not None:

        bulk_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Employee Data")
        st.dataframe(bulk_data, use_container_width=True)
        st.caption(f"Total rows: {len(bulk_data):,}")

        if st.button("🚀 Predict Uploaded Employees"):

            missing_cols = [c for c in features if c not in bulk_data.columns]

            if missing_cols:
                st.error(f"❌ Missing Columns: {missing_cols}")

            else:
                bulk_X      = bulk_data[features]
                bulk_scaled = scaler.transform(bulk_X)
                bulk_preds  = model.predict(bulk_scaled)
                bulk_probas = model.predict_proba(bulk_scaled)

                def row_reasons(row):
                    r = get_reasons(row.to_dict())
                    return ", ".join([label for label, _ in r])

                bulk_data['Prediction']            = ["WILL LEAVE" if p==1 else "WILL STAY" for p in bulk_preds]
                bulk_data['Leave Probability (%)'] = (bulk_probas[:,1]*100).round(2)
                bulk_data['Reasons']               = bulk_data[features].apply(row_reasons, axis=1)

                n_leave = (bulk_preds == 1).sum()
                n_stay  = (bulk_preds == 0).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Employees", len(bulk_preds))
                c2.metric("🔴 Will Leave",   int(n_leave))
                c3.metric("🟢 Will Stay",    int(n_stay))

                st.subheader("Prediction Results")
                st.dataframe(
                    bulk_data.style.map(
                        lambda v: "background-color:#ffd6d6;color:#c0392b;font-weight:bold;"
                        if v == "WILL LEAVE" else
                        ("background-color:#d6f5e3;color:#1e8449;font-weight:bold;"
                         if v == "WILL STAY" else ""),
                        subset=["Prediction"]
                    ),
                    use_container_width=True
                )

                csv_out = bulk_data.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download Results CSV",
                    csv_out,
                    "employee_attrition_results.csv",
                    "text/csv"
                )


# ─── FOOTER ──────────────────────────────────────────────
st.markdown("---")
st.caption("Developed using Streamlit + XGBoost + SMOTE")
