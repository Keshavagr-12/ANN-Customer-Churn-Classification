import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import streamlit as st
# import tensorflow as tf

# from tensorflow.keras.models import load_model
# model = tf.keras.models.load_model('model.h5')

import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
one_hot_encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))

with open('label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder = pickle.load(file)

with open ('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


st.title('Customer churn prediction')

geography = st.selectbox('Geography',one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])



input_df = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [label_encoder.transform([gender])[0]],
'Geography':[geography], 
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})


geo_encoded = one_hot_encoder.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))
input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)
input_scaled = scaler.transform(input_df)



if st.button('Predict Churn'):
    with st.spinner('Calculating...'):
        prediction = model.predict(input_scaled)
        prediction_proba = float(prediction[0][0])
        
    st.subheader('Results:')
    st.write(f"**Churn Probability:** {prediction_proba:.2%}")
    
    if prediction_proba > 0.5:
        st.error('⚠️ **The customer is likely to churn.**')
    else:
        st.success('✅ **The customer is not likely to churn.**')




# import logging
# import os
# import pickle

# # 1. SILENCE TENSORFLOW WARNINGS COMPLETELY (Place before other imports)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging.getLogger('absl').setLevel(logging.ERROR)

# import pandas as pd
# import streamlit as st
# import tensorflow as tf

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="Churn Guardian",
#     page_icon="🔮",
#     layout="centered"
# )

# # --- CACHED ASSET LOADING ---
# @st.cache_resource
# def load_ml_assets():
#     """Loads and compiles ML components once, caching them in memory."""
#     # Load model and compile programmatically to suppress metric warnings
#     model = tf.keras.models.load_model('model.h5')
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
#     with open('label_encoder.pkl', 'rb') as file:
#         label_encoder = pickle.load(file)
        
#     with open('one_hot_encoder.pkl', 'rb') as file:
#         one_hot_encoder = pickle.load(file)
        
#     with open('scaler.pkl', 'rb') as file:
#         scaler = pickle.load(file)
        
#     return model, label_encoder, one_hot_encoder, scaler

# # Initialize assets
# model, label_encoder, one_hot_encoder, scaler = load_ml_assets()


# # --- UI HEADER ---
# st.title('🔮 Customer Churn Prediction')
# st.markdown("Assess the probability of a customer leaving your platform using our Deep Learning model.")
# st.divider()


# # --- TABBED USER INPUT LAYOUT ---
# # Splitting inputs into logical categories makes the form much cleaner to fill out
# tab1, tab2 = st.tabs(["📊 Demographics & Account", "💳 Behavioral Metrics"])

# with tab1:
#     st.subheader("Profile Information")
#     col1, col2 = st.columns(2)
#     with col1:
#         geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
#         gender = st.selectbox('Gender', label_encoder.classes_)
#     with col2:
#         age = st.slider('Age', min_value=18, max_value=92, value=35)
#         credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
        
#     st.subheader("Financial Standing")
#     balance = st.number_input('Account Balance ($)', min_value=0.0, value=0.0, step=500.0)
#     estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0)

# with tab2:
#     st.subheader("Engagement & Tools")
#     col3, col4 = st.columns(2)
#     with col3:
#         tenure = st.slider('Tenure (Years)', 0, 10, 5)
#         num_of_products = st.slider('Number of Products Subscribed', 1, 4, 1)
#     with col4:
#         has_cr_card = st.radio('Has Credit Card?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
#         is_active_member = st.radio('Is Active Member?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)


# # --- DATA PREPROCESSING ENGINE ---
# # 1. Transform basic features
# input_df = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder.transform([gender])[0]],
#     'Geography': [geography], 
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary]
# })

# # 2. Extract and match One-Hot Encoded features
# geo_encoded = one_hot_encoder.transform(input_df[['Geography']]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# # 3. Drop base geography and stitch back together
# input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

# # 💡 CRITICAL STABILITY CHECK:
# # If your model was trained with 'Geography' encoded columns at the front or specific indexes,
# # explicitly rearrange columns right here to match your `X_train.columns` configuration.
# # example: input_df = input_df[['CreditScore', 'Geography_France', 'Geography_Germany', ...]]

# # 4. Final Feature Scaling
# input_scaled = scaler.transform(input_df)


# # --- PREDICTION WORKFLOW ---
# st.divider()
# if st.button('Run Risk Analysis', type='primary', use_container_width=True):
#     with st.spinner('Analyzing consumer telemetry...'):
#         prediction = model.predict(input_scaled)
#         prediction_proba = float(prediction[0][0])
        
#     st.subheader('Analysis Results')
    
#     # Visual Progress Indicator
#     st.progress(prediction_proba)
    
#     # Conditional Response Display
#     if prediction_proba > 0.5:
#         st.error(f'⚠️ **High Risk Alert:** The customer has a **{prediction_proba:.2%}** probability of churning.')
        
#         # Actionable insights metrics based on input flags
#         if is_active_member == 0:
#             st.info("💡 *Recommendation: Initiate re-engagement marketing. This customer is classified as an inactive member.*")
#     else:
#         st.success(f'✅ **Low Risk Retention:** The customer is stable with only a **{prediction_proba:.2%}** churn probability.')