import streamlit as st
import pandas as pd
import joblib
import json
from PIL import Image

# -------------------------
# Page Config (MUST be first Streamlit call)
# -------------------------
st.set_page_config(
    page_title="Hotel Haven",
    layout="centered"
)

# -------------------------
# Header + Logo
# -------------------------
try:
    logo = Image.open("Hotel Haven.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=200)
except Exception:
    # If logo file isn't found, app still runs
    pass

st.markdown("""
<div style='text-align:center; padding:10px'>
<h4 style='color:grey;'>AI-Powered Booking Cancellation Risk Dashboard</h4>
</div>
""", unsafe_allow_html=True)

st.divider()

st.write(
    "Complete the booking form below to estimate cancellation risk and receive operational guidance."
)

# -------------------------
# Load model + feature columns
# -------------------------
model = joblib.load("hotel_haven_rf_pipeline.joblib")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# -------------------------
# Dropdown options (from your dataset)
# -------------------------
meal_options = ["Meal Plan 1", "Meal Plan 2", "Not Selected"]
room_options = ["Room_Type 1", "Room_Type 2", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7", "Other"]
market_options = ["Corporate", "Offline", "Online", "Other"]

months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# NOTE: Your dataset has customer_profile as numeric codes.
# We hide the "code" language and present staff-friendly labels.
guest_type_map = {
    "New guest / first-time booking": 0,
    "Returning guest": 1,
    "Business traveller": 2,
    "Family booking": 3,
    "Group booking": 4,
    "Other / Unknown": 13,
}
# If you want to include ALL codes exactly:
# guest_type_map = {
#     "Profile 0": 0, "Profile 1": 1, "Profile 2": 2, "Profile 3": 3, "Profile 4": 4,
#     "Profile 5": 5, "Profile 6": 6, "Profile 11": 11, "Profile 13": 13
# }

# -------------------------
# Form (better UX + question-based labels)
# -------------------------
with st.form("booking_form"):
    st.subheader("Guest and Stay Details")

    number_of_adults = st.number_input(
        "How many adults are included in this booking?",
        min_value=0, max_value=10, value=2, step=1
    )
    number_of_children = st.number_input(
        "How many children are included in this booking?",
        min_value=0, max_value=10, value=0, step=1
    )

    colA, colB = st.columns(2)
    with colA:
        number_of_week_nights = st.number_input(
            "How many weekday nights will the guest stay? (Mon–Thu)",
            min_value=0, max_value=30, value=2, step=1
        )
    with colB:
        number_of_weekend_nights = st.number_input(
            "How many weekend nights will the guest stay? (Fri–Sun)",
            min_value=0, max_value=30, value=1, step=1
        )
    total_nights = number_of_week_nights + number_of_weekend_nights

    st.write(f"Total stay length: **{total_nights} nights**")

    st.subheader("Booking Preferences")

    type_of_meal = st.selectbox(
        "Which meal plan is selected for this booking?",
        meal_options
    )
    room_type = st.selectbox(
        "Which room type is being booked?",
        room_options
    )

    colC, colD = st.columns(2)
    with colC:
        car_parking_choice = st.selectbox(
            "Will the guest require a parking space?",
            ["No", "Yes"]
        )
    with colD:
        repeated_choice = st.selectbox(
            "Is this a returning guest?",
            ["No", "Yes"]
        )

    car_parking_space = 1 if car_parking_choice == "Yes" else 0
    repeated_val = 1 if repeated_choice == "Yes" else 0

    lead_time = st.number_input(
        "How many days in advance was this booking made? (Lead time)",
        min_value=0, max_value=365, value=30, step=1
    )

    st.subheader("Guest Type and Booking Channel")

    guest_type_label = st.selectbox(
        "Which option best describes the guest type for this booking?",
        list(guest_type_map.keys())
    )
    customer_profile = guest_type_map[guest_type_label]

    market_segment = st.selectbox(
        "How was this booking made?",
        market_options,
        help="Example: Online, Offline, Corporate, Other."
    )

    st.subheader("Price and Requests")

    average_price = st.number_input(
        "What is the average nightly price (£) for this booking?",
        min_value=30, max_value=800, value=120, step=5,
        help="Enter a realistic nightly price to improve prediction quality."
    )

    special_requests = st.number_input(
        "How many special requests were made for this booking?",
        min_value=0, max_value=10, value=0, step=1
    )

    st.subheader("Reservation Timing")

    reservation_year = st.number_input(
        "Reservation year",
        min_value=2015, max_value=2035, value=2026, step=1
    )

    month_name = st.selectbox("Reservation month", months)
    reservation_month = months.index(month_name) + 1  # 1–12

    day_name = st.selectbox("Reservation day of the week", days)
    reservation_dayofweek = days.index(day_name)  # 0–6 (matches your dataset)

    submitted = st.form_submit_button("Predict Cancellation Risk")

# -------------------------
# Derived features (hidden from user)
# -------------------------
special_requests_binary = 1 if special_requests > 0 else 0
non_customer_profile_binary = 1 if customer_profile == 0 else 0

# -------------------------
# Predict + Staff-friendly output
# -------------------------
if submitted:  # or: if st.button("Predict Cancellation Risk"):

    # Build input frame
    input_data = {
        "number of adults": number_of_adults,
        "number of children": number_of_children,
        "number of weekend nights": number_of_weekend_nights,
        "number of week nights": number_of_week_nights,
        "type of meal": type_of_meal,
        "car parking space": car_parking_space,
        "room type": room_type,
        "lead time": lead_time,
        "repeated": repeated_val,
        "customer_profile": customer_profile,  
        "average price": average_price,
        "special requests": special_requests,
        "market_segment": market_segment,
        "non_customer_profile_binary": non_customer_profile_binary,
        "special_requests_binary": special_requests_binary,
        "reservation_year": reservation_year,
        "reservation_month": reservation_month,
        "reservation_dayofweek": reservation_dayofweek,
    }

    df_input = pd.DataFrame([input_data]).reindex(columns=feature_columns)

    # Predict
    proba = float(model.predict_proba(df_input)[0][1])
    risk_pct = proba * 100

    total_guests = int(number_of_adults + number_of_children)
    total_nights = int(number_of_week_nights + number_of_weekend_nights)

    st.divider()

    # -------------------------
    # Booking Summary 
    # -------------------------
    st.subheader("Booking Summary")
    st.write(
        f"""
- **Total guests:** {total_guests}
- **Room type:** {room_type}
- **Total stay:** {total_nights} nights
- **Average nightly price:** £{int(average_price)}
- **Market segment:** {market_segment}
"""
    )

    st.divider()

    # -------------------------
    # Cancellation Risk Assessment (staff-friendly)
    # -------------------------
    st.subheader("Cancellation Risk Assessment")

    if proba < 0.30:
        st.success("✅ Low cancellation risk")
        summary = (
            f"This booking looks stable. Our model estimates a **{risk_pct:.1f}%** chance of cancellation."
        )
        guidance = [
            "Proceed with standard confirmation procedures.",
            "Optional: offer add-ons (meal plan, parking) to improve guest commitment.",
        ]
        dot = "🟢"
        colour = "#2ECC71"
    elif proba < 0.70:
        st.warning("⚠️ Moderate cancellation risk")
        summary = (
            f"This booking has a noticeable chance of cancellation. Our model estimates a **{risk_pct:.1f}%** chance of cancellation."
        )
        guidance = [
            "Send a reminder closer to arrival (email/SMS).",
            "Consider reconfirming payment details if policy permits.",
            "Monitor inventory closely if demand is high.",
        ]
        dot = "🟠"
        colour = "#F1C40F"
    else:
        st.error("🚨 High cancellation risk")
        summary = (
            f"This booking is more likely to be cancelled based on historical patterns. Our model estimates a **{risk_pct:.1f}%** chance of cancellation."
        )
        guidance = [
            "Consider requesting a deposit / confirmation of payment (if policy permits).",
            "Send a confirmation call/email within 24–48 hours.",
            "Review inventory strategy (waitlist/overbooking rules) where applicable.",
        ]
        dot = "🔴"
        colour = "#E74C3C"

    st.write(summary)

    # Colour-changing “BI style” risk score (this MUST be inside submitted)
    st.markdown(
        f"<h3 style='color:{colour};'>{dot} Risk Score: {risk_pct:.1f}%</h3>",
        unsafe_allow_html=True,
    )

    st.caption("Estimated cancellation risk")
    st.progress(min(max(proba, 0.0), 1.0))  # progress expects 0–1

    st.markdown("#### Recommended Next Steps")
    for g in guidance:
        st.write(f"• {g}")

    with st.expander("How should staff interpret this result?"):
        st.write(
            """
This tool gives a probability estimate based on past booking patterns.
Use it for decision support — not as a guarantee of cancellation.

Higher risk scores suggest the booking resembles previous cancellations (e.g., long lead time, certain segments, fewer commitments).
Use this to guide proactive actions and manage inventory risk.
"""
        )
# -------------------------
# About section (cleaner, staff-facing)
# -------------------------
st.divider()
st.subheader("About This Tool")
st.write(
    """
    This application supports hotel operations by estimating the likelihood of booking cancellation.
    It uses a machine learning model trained on historical Hotel Haven booking data and evaluates
    factors such as lead time, length of stay, room type, market segment, special requests, and timing.

    Use the prediction to support decision-making around confirmations, deposits, reminders, and inventory planning.
    """
      )