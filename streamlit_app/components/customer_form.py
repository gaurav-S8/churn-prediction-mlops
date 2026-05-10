import streamlit as st

def customer_form(key_prefix=""):
    with st.form(f"form_{key_prefix}"):
        st.markdown('<div class = "card-label">Customer Profile</div>', unsafe_allow_html = True)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            customer_id = st.text_input("Customer ID", value = "C001", key = f"{key_prefix}_id")
            gender = st.selectbox("Gender", ["Male", "Female"], key = f"{key_prefix}_gender")
            senior = st.selectbox("Senior Citizen", [0, 1], key = f"{key_prefix}_senior")
            partner = st.selectbox("Partner", ["Yes", "No"], key = f"{key_prefix}_partner")
            dependents = st.selectbox("Dependents", ["Yes", "No"], key = f"{key_prefix}_dep")

        with c2:
            tenure = st.number_input("Tenure (months)", 0, 3000, 12, key = f"{key_prefix}_tenure")
            phone = st.selectbox("Phone Service", ["Yes", "No"], key = f"{key_prefix}_phone")
            multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key = f"{key_prefix}_ml")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key = f"{key_prefix}_int")
            online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key = f"{key_prefix}_osec")

        with c3:
            online_bk = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key = f"{key_prefix}_obk")
            device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key = f"{key_prefix}_dp")
            tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key = f"{key_prefix}_ts")
            stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key = f"{key_prefix}_stv")
            stream_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key = f"{key_prefix}_smv")

        with c4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key = f"{key_prefix}_contract")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key = f"{key_prefix}_pb")
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ], key = f"{key_prefix}_pay")
            monthly = st.number_input("Monthly Charges", 0.0, value = 50.0, key = f"{key_prefix}_mc")
            total = st.number_input("Total Charges", 0.0, value = 600.0, key = f"{key_prefix}_tc")

        submitted = st.form_submit_button("Run Inference", use_container_width = True)

    if submitted:
        return {
            "CustomerID": customer_id, "Gender": gender, "SeniorCitizen": int(senior),
            "Partner": partner, "Dependents": dependents, "Tenure": int(tenure),
            "PhoneService": phone, "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bk, "DeviceProtection": device_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": float(monthly), "TotalCharges": float(total)
        }
    return None