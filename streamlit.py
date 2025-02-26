import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load pre-trained models and scalers
with open('scaler.pkl', 'rb') as f:
    scaler_clf = pickle.load(f)
with open('scaler_regg.pkl', 'rb') as f:
    scaler_reg = pickle.load(f)
with open('scaler_cluster.pkl', 'rb') as f:
    scaler_cluster = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('DecisionTree.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('xgbregressor.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('kmeans.pkl', 'rb') as f:
    cluster_model = pickle.load(f)

# Top 10 most frequent values from training data
top_10_models = ['B4', 'A2', 'A11', 'P1', 'B10', 'A4', 'A15', 'A5', 'A10', 'A1']

def preprocess_data(df, task):
    # Drop unwanted columns
    df = df.drop(columns=['year', 'month', 'day', 'order', 'country', 'session_id'], errors='ignore')
    
    # Encode 'page2_clothing_model'
    df['page2_clothing_model'] = df['page2_clothing_model'].apply(lambda x: x if x in top_10_models else 'others')
    encoded_features = encoder.transform(df[['page2_clothing_model']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['page2_clothing_model']))
    df = pd.concat([df.drop(columns=['page2_clothing_model']), encoded_df], axis=1)
    
    # Scale data
    if task == 'Regression':
        df_scaled = scaler_reg.transform(df.drop(columns=['price'], errors='ignore'))
    elif task == 'Classification':
        df_scaled = scaler_clf.transform(df.drop(columns=['price_2'], errors='ignore'))
    else:  # Clustering
        df_scaled = scaler_cluster.transform(df)
    
    return df_scaled

def main():
    st.title("Customer Conversion Analysis")
    st.markdown("---")

    st.write("""Created by:
             KISHORE KUMAR B N""")
    st.write("' This project focusses on Customer Conversion Analysis for Online Shopping Using Clickstream Data '")
    st.write(' NOTE : ')
    st.write("' ** In this streamlit app the user can either choose to upload his/her own csv file or there is a cusion to manually enter the available data and get the desired output ** '")
    st.write("' ** Before uploading your csv file or manually entering data, select the desired function that needs to be performed from the dropdown below . **'")
    st.markdown("---")
    
    # Select task
    task = st.selectbox("Select Task", ["Regression", "Classification", "Clustering"])
    
    # Choose input method
    input_method = st.radio("How would you like to input data?", ["Upload CSV", "Enter Manually"])
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df_processed = preprocess_data(df, task)
            
            # Predict
            if task == "Regression":
                predictions = reg_model.predict(df_processed)
                df['Predicted Price'] = predictions
            elif task == "Classification":
                predictions = clf_model.predict(df_processed)
                df['Predicted Price Category'] = predictions
            else:
                predictions = cluster_model.predict(df_processed)
                df['Cluster'] = predictions
            
            st.write("Prediction Results:")
            st.dataframe(df)
    else:
        st.write("Enter data manually:")

        if task in ["Regression", "Classification"]:
            manual_input = {}

            manual_input['page1_main_category'] = st.selectbox("Select Page 1 Main Category : where 1 -trousers 2-skirts 3-blouses 4-sale", [1,2,3,4])
            manual_input['page2_clothing_model'] = st.selectbox("Select Page 2 Clothing Model : The mentoned ae the famous (Click 'others' if u have a different option)", top_10_models + ['others'])
            manual_input['colour'] = st.text_input("Enter Colour (1-14) ,Each number represent different colour")
            manual_input['location'] = st.selectbox("Select Location where : 1-top left, 2-top in the middle, 3-top right,4-bottom left,  5-bottom in the middle,  6-bottom right", [1,2,3,4,5,6])
            manual_input['model_photography'] = st.selectbox("Select Model Photography where : 1-en face ,2-profile", [1,2])
            manual_input['page'] = st.selectbox("Select Page : (page number within the e-store website)", [1, 2, 3, 4, 5])

            df = pd.DataFrame([manual_input])
            df_processed = preprocess_data(df, task)

            if st.button("Predict"):
                if task == "Regression":
                    prediction = reg_model.predict(df_processed)[0]
                    st.write(f"Predicted Price: {prediction}")
                else:  # Classification
                    prediction = clf_model.predict(df_processed)[0]
                    st.write(f"Predicted Price Category: {prediction}")

        elif task == "Clustering":
            manual_input = {}

            manual_input['page1_main_category'] = st.selectbox("Select Page 1 Main Category : where 1 -trousers 2-skirts 3-blouses 4-sale", [1,2,3,4])
            manual_input['colour'] = st.text_input("Enter Colour (1-14) ,Each number represent different colour")
            manual_input['location'] = st.selectbox("Select Location where : 1-top left, 2-top in the middle, 3-top right,4-bottom left,  5-bottom in the middle,  6-bottom right", [1,2,3,4,5,6])
            manual_input['model_photography'] = st.selectbox("Select Model Photography where : 1-en face ,2-profile", [1,2])
            manual_input['price'] = st.number_input("Enter Price (Enter Value from 20-100)", min_value=0.0, format="%.2f")
            manual_input['price_2'] = st.selectbox("Select Price 2 : the average price for the entire product category 1-yes,2-no", [1,2])
            manual_input['page'] = st.selectbox("Select Page : (page number within the e-store website)", [1, 2, 3, 4, 5])
            manual_input['page2_clothing_model'] = st.selectbox("Select Page 2 Clothing Model : The mentoned ae the famous (Click 'others' if u have a different option)", top_10_models + ['others'])


            df = pd.DataFrame([manual_input])
            df_processed = preprocess_data(df, task)

            if st.button("Predict"):
                prediction = cluster_model.predict(df_processed)[0]
                st.write(f"Cluster: {prediction}")

if __name__ == "__main__":
    main()
