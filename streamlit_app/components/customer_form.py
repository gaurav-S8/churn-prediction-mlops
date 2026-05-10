# Import Libraries
import json
import random
import pyperclip
import streamlit as st

def random_payload():
    return {
        "CustomerID": f"C{random.randint(100, 999)}",
        "Gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "Tenure": random.randint(0, 72),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
        "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
        "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
        "DeviceProtection": random.choice(["Yes", "No", "No internet service"]),
        "TechSupport": random.choice(["Yes", "No", "No internet service"]),
        "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
        "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
        "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ]),
        "MonthlyCharges": round(random.uniform(20, 120), 2),
        "TotalCharges": round(random.uniform(20, 8000), 2)
    }

def get_index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0

def customer_form(key_prefix = "", mode = "predict"):

    # Load defaults from session state if randomized
    d = st.session_state.get(f"{key_prefix}_data", {})

    gender_opts = ["Male", "Female"]
    yesno_opts = ["Yes", "No"]
    lines_opts = ["Yes", "No", "No phone service"]
    internet_opts = ["DSL", "Fiber optic", "No"]
    internet_svc_opts = ["Yes", "No", "No internet service"]
    contract_opts = ["Month-to-month", "One year", "Two year"]
    payment_opts = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    senior_opts = [0, 1]

    with st.form(f"form_{key_prefix}"):
        # Action Bar
        if mode == "predict":
            _, a1, a2 = st.columns([10, 2, 2])

            with a1:
                random_clicked = st.form_submit_button(
                    "Randomize",
                    use_container_width = True
                )

            with a2:
                copy_clicked = st.form_submit_button(
                    "Copy Customer Profile",
                    use_container_width = True
                )
        elif mode == "explain":
            _, a1 = st.columns([10, 2])

            with a1:
                paste_info_clicked = st.form_submit_button(
                    "Paste Customer Profile",
                    use_container_width = True
                )

        # Randomize Logic
        if mode == "predict" and random_clicked:
            payload = random_payload()
            st.session_state[f"{key_prefix}_data"] = payload

            # Update widget states
            st.session_state[f"{key_prefix}_id"] = payload["CustomerID"]
            st.session_state[f"{key_prefix}_gender"] = payload["Gender"]
            st.session_state[f"{key_prefix}_senior"] = payload["SeniorCitizen"]
            st.session_state[f"{key_prefix}_partner"] = payload["Partner"]
            st.session_state[f"{key_prefix}_dep"] = payload["Dependents"]

            st.session_state[f"{key_prefix}_tenure"] = payload["Tenure"]
            st.session_state[f"{key_prefix}_phone"] = payload["PhoneService"]
            st.session_state[f"{key_prefix}_ml"] = payload["MultipleLines"]
            st.session_state[f"{key_prefix}_int"] = payload["InternetService"]
            st.session_state[f"{key_prefix}_osec"] = payload["OnlineSecurity"]

            st.session_state[f"{key_prefix}_obk"] = payload["OnlineBackup"]
            st.session_state[f"{key_prefix}_dp"] = payload["DeviceProtection"]
            st.session_state[f"{key_prefix}_ts"] = payload["TechSupport"]
            st.session_state[f"{key_prefix}_stv"] = payload["StreamingTV"]
            st.session_state[f"{key_prefix}_smv"] = payload["StreamingMovies"]

            st.session_state[f"{key_prefix}_contract"] = payload["Contract"]
            st.session_state[f"{key_prefix}_pb"] = payload["PaperlessBilling"]
            st.session_state[f"{key_prefix}_pay"] = payload["PaymentMethod"]

            st.session_state[f"{key_prefix}_mc"] = payload["MonthlyCharges"]
            st.session_state[f"{key_prefix}_tc"] = payload["TotalCharges"]

            st.rerun()
        
        # Copy Customer Profile Logic
        if mode == "predict" and copy_clicked:
            current_payload = {

                "CustomerID": st.session_state.get(f"{key_prefix}_id"),
                "Gender": st.session_state.get(f"{key_prefix}_gender"),
                "SeniorCitizen": st.session_state.get(f"{key_prefix}_senior"),
                "Partner": st.session_state.get(f"{key_prefix}_partner"),
                "Dependents": st.session_state.get(f"{key_prefix}_dep"),

                "Tenure": st.session_state.get(f"{key_prefix}_tenure"),
                "PhoneService": st.session_state.get(f"{key_prefix}_phone"),
                "MultipleLines": st.session_state.get(f"{key_prefix}_ml"),
                "InternetService": st.session_state.get(f"{key_prefix}_int"),
                "OnlineSecurity": st.session_state.get(f"{key_prefix}_osec"),

                "OnlineBackup": st.session_state.get(f"{key_prefix}_obk"),
                "DeviceProtection": st.session_state.get(f"{key_prefix}_dp"),
                "TechSupport": st.session_state.get(f"{key_prefix}_ts"),
                "StreamingTV": st.session_state.get(f"{key_prefix}_stv"),
                "StreamingMovies": st.session_state.get(f"{key_prefix}_smv"),

                "Contract": st.session_state.get(f"{key_prefix}_contract"),
                "PaperlessBilling": st.session_state.get(f"{key_prefix}_pb"),
                "PaymentMethod": st.session_state.get(f"{key_prefix}_pay"),

                "MonthlyCharges": st.session_state.get(f"{key_prefix}_mc"),
                "TotalCharges": st.session_state.get(f"{key_prefix}_tc")
            }

            # Save in session memory
            st.session_state[f"{key_prefix}_copied_profile"] = current_payload

            # Convert to JSON
            payload_json = json.dumps(current_payload, indent = 2)
            # Copy to clipboard
            pyperclip.copy(payload_json)
            st.toast("Customer profile copied to clipboard")
        
        if mode == "explain" and paste_info_clicked:
            payload = st.session_state.get("predict_copied_profile")
            print(payload)
            if payload:

                st.session_state[f"{key_prefix}_data"] = payload

                st.session_state[f"{key_prefix}_id"] = payload["CustomerID"]
                st.session_state[f"{key_prefix}_gender"] = payload["Gender"]
                st.session_state[f"{key_prefix}_senior"] = payload["SeniorCitizen"]
                st.session_state[f"{key_prefix}_partner"] = payload["Partner"]
                st.session_state[f"{key_prefix}_dep"] = payload["Dependents"]

                st.session_state[f"{key_prefix}_tenure"] = payload["Tenure"]
                st.session_state[f"{key_prefix}_phone"] = payload["PhoneService"]
                st.session_state[f"{key_prefix}_ml"] = payload["MultipleLines"]
                st.session_state[f"{key_prefix}_int"] = payload["InternetService"]
                st.session_state[f"{key_prefix}_osec"] = payload["OnlineSecurity"]

                st.session_state[f"{key_prefix}_obk"] = payload["OnlineBackup"]
                st.session_state[f"{key_prefix}_dp"] = payload["DeviceProtection"]
                st.session_state[f"{key_prefix}_ts"] = payload["TechSupport"]
                st.session_state[f"{key_prefix}_stv"] = payload["StreamingTV"]
                st.session_state[f"{key_prefix}_smv"] = payload["StreamingMovies"]

                st.session_state[f"{key_prefix}_contract"] = payload["Contract"]
                st.session_state[f"{key_prefix}_pb"] = payload["PaperlessBilling"]
                st.session_state[f"{key_prefix}_pay"] = payload["PaymentMethod"]

                st.session_state[f"{key_prefix}_mc"] = payload["MonthlyCharges"]
                st.session_state[f"{key_prefix}_tc"] = payload["TotalCharges"]

                st.toast("Customer profile pasted")
                st.rerun()
            else:
                st.warning("No copied customer profile found")

        st.markdown('<div class="card-label">Customer Profile</div>', unsafe_allow_html = True)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            customer_id = st.text_input(
                "Customer ID",
                value = d.get("CustomerID", "C001"),
                key = f"{key_prefix}_id"
            )
            gender = st.selectbox(
                "Gender", gender_opts,
                index = get_index(gender_opts, d.get("Gender", "Male")),
                key = f"{key_prefix}_gender"
            )
            senior = st.selectbox(
                "Senior Citizen", senior_opts,
                index = get_index(senior_opts, d.get("SeniorCitizen", 0)),
                key = f"{key_prefix}_senior"
            )
            partner = st.selectbox(
                "Partner", yesno_opts,
                index = get_index(yesno_opts, d.get("Partner", "Yes")),
                key = f"{key_prefix}_partner"
            )
            dependents = st.selectbox(
                "Dependents", yesno_opts,
                index = get_index(yesno_opts, d.get("Dependents", "No")),
                key = f"{key_prefix}_dep"
            )

        with c2:
            tenure = st.number_input(
                "Tenure (months)", 0, 3000, d.get("Tenure", 12), key = f"{key_prefix}_tenure"
            )
            phone = st.selectbox(
                "Phone Service", yesno_opts,
                index = get_index(yesno_opts, d.get("PhoneService", "Yes")),
                key = f"{key_prefix}_phone"
            )
            multi_lines = st.selectbox(
                "Multiple Lines", lines_opts,
                index = get_index(lines_opts, d.get("MultipleLines", "No")),
                key = f"{key_prefix}_ml"
            )
            internet = st.selectbox(
                "Internet Service", internet_opts,
                index = get_index(internet_opts, d.get("InternetService", "DSL")),
                key = f"{key_prefix}_int"
            )
            online_sec = st.selectbox(
                "Online Security", internet_svc_opts,
                index = get_index(internet_svc_opts, d.get("OnlineSecurity", "Yes")),
                key = f"{key_prefix}_osec"
            )

        with c3:
            online_bk = st.selectbox(
                "Online Backup", internet_svc_opts,
                index = get_index(internet_svc_opts, d.get("OnlineBackup", "No")),
                key = f"{key_prefix}_obk"
            )
            device_prot = st.selectbox(
                "Device Protection", internet_svc_opts,
                index = get_index(internet_svc_opts, d.get("DeviceProtection", "Yes")),
                key = f"{key_prefix}_dp"
            )
            tech_sup = st.selectbox(
                "Tech Support", internet_svc_opts,
                index = get_index(internet_svc_opts, d.get("TechSupport", "No")),
                key = f"{key_prefix}_ts"
            )
            stream_tv = st.selectbox(
                "Streaming TV", internet_svc_opts,
                index = get_index(internet_svc_opts, d.get("StreamingTV", "Yes")),
                key = f"{key_prefix}_stv"
            )
            stream_mv = st.selectbox(
                "Streaming Movies", internet_svc_opts,
                index = get_index(internet_svc_opts, d.get("StreamingMovies", "No")),
                key = f"{key_prefix}_smv"
            )

        with c4:
            contract = st.selectbox(
                "Contract", contract_opts,
                index = get_index(contract_opts, d.get("Contract", "Month-to-month")),
                key = f"{key_prefix}_contract"
            )
            paperless = st.selectbox(
                "Paperless Billing", yesno_opts,
                index = get_index(yesno_opts, d.get("PaperlessBilling", "Yes")),
                key = f"{key_prefix}_pb"
            )
            payment = st.selectbox(
                "Payment Method", payment_opts,
                index = get_index(payment_opts, d.get("PaymentMethod", "Electronic check")),
                key = f"{key_prefix}_pay"
            )
            monthly = st.number_input(
                "Monthly Charges", 0.0,
                value = float(d.get("MonthlyCharges", 50.0)),
                key = f"{key_prefix}_mc"
            )
            total = st.number_input(
                "Total Charges", 0.0,
                value = float(d.get("TotalCharges", 600.0)),
                key = f"{key_prefix}_tc"
            )

        submitted = st.form_submit_button(
            "Run Inference", use_container_width = True
        )

    if submitted:
        payload = {
            "CustomerID": customer_id, "Gender": gender, "SeniorCitizen": int(senior),
            "Partner": partner, "Dependents": dependents, "Tenure": int(tenure),
            "PhoneService": phone, "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bk, "DeviceProtection": device_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": float(monthly), "TotalCharges": float(total)
        }
        # Save to session state for cross-page use
        st.session_state[f"{key_prefix}_data"] = payload
        return payload
    return None