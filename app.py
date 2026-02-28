import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
# ---------------- PAGE CONFIG (ONLY ONCE & AT TOP) ----------------
st.set_page_config(
page_title="UPI Transaction Analytics",
  page_icon="💳",
layout="wide"
)
# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
background: #2A7B9B;
background: linear-gradient(5deg,rgba(42, 123, 155, 1) 0%, rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: #11ED6C
.stButton>button {
    background-color: #0066cc;
    color: white;
    font-size: 16px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE SECTION ----------------
st.title("📊 UPI Transaction Analytics & Expense Prediction")
st.write("Data Analytics & Machine Learning Based Expense Prediction")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to:",
    ["Dashboard", "Prediction Model"]
)

# Load CSV
df = pd.read_csv("MyTransaction.csv")

# Cleaning
df['Date'] = pd.to_datetime(df['Date.1'], dayfirst=True, errors='coerce')
df['Withdrawal'] = df['Withdrawal'].fillna(0)
df['Deposit'] = df['Deposit'].fillna(0)
df = df.drop(columns=['Date.1'])

debit_df = df[df['Withdrawal'] > 0]
# Monthly calculation
debit_df['Month'] = debit_df['Date'].dt.month
monthly = debit_df.groupby('Month')['Withdrawal'].sum().reset_index()
month_names = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
}

monthly['Month_Name'] = monthly['Month'].map(month_names)



# ---------------- DASHBOARD SECTION ----------------
if option == "Dashboard":

    st.subheader("📊 Financial Overview")

    total_spent = debit_df['Withdrawal'].sum()
    total_received = df['Deposit'].sum()
    avg_spent = debit_df['Withdrawal'].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Spent", f"₹{round(total_spent,2)}")
    col2.metric("Total Received", f"₹{round(total_received,2)}")
    col3.metric("Average Expense", f"₹{round(avg_spent,2)}")

    st.divider()
   
    # Line Chart
    st.subheader("📈 Monthly Spending Trend")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(monthly['Month_Name'], monthly['Withdrawal'], marker='o')
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount Spent")
    ax.grid(True)
    st.pyplot(fig,clear_figure=True)

    st.divider()
   
    # Pie Chart
    st.subheader("📊 Expense Distribution (Spent vs Received)")
    pie_data = [total_spent, total_received]
    labels = ["Total Spent", "Total Received"]

    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.pie(pie_data, labels=labels, autopct='%1.1f%%')
    ax2.axis('equal')
    st.pyplot(fig2,clear_figure=True)

    st.divider()

    # Download Button
    st.subheader("📥 Download Monthly Report")
    csv = monthly.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report as CSV",
        data=csv,
        file_name='monthly_report.csv',
        mime='text/csv'
    )


# ---------------- PREDICTION SECTION ----------------
elif option == "Prediction Model":

    st.subheader("🤖 Expense Prediction Model")

    X = monthly[['Month']]
    y = monthly['Withdrawal']

    model = LinearRegression()
    model.fit(X, y)

    # Select month name
    selected_month_name = st.selectbox(
        "Select Month",
        list(month_names.values())
    )

    # Convert month name to month number
    selected_month = list(month_names.keys())[ 
        list(month_names.values()).index(selected_month_name)
    ]

    if st.button("Predict Expense"):

        prediction = model.predict([[selected_month]])

        st.success(
            f"Predicted Expense for {selected_month_name}: ₹{round(prediction[0], 2)}"
        )

        # Show prediction graph
        fig3, ax3 = plt.subplots(figsize=(6,3))
        ax3.plot(monthly['Month'], monthly['Withdrawal'], marker='o')
        ax3.scatter(selected_month, prediction[0],s=150)
        ax3.set_xlabel("Month Number")
        ax3.set_ylabel("Amount")
        ax3.grid(True)      
        st.pyplot(fig3,clear_figure=True)
        st.markdown("---")
        st.markdown(
        "<center>© 2026 UPI Analytics Dashboard | Developed by Sahaj Kaur</center>",
        unsafe_allow_html=True
)

