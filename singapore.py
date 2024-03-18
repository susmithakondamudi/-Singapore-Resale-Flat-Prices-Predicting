# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import warnings
warnings.simplefilter("ignore")
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")
def setting_bg(background_image_url):
        st.markdown(f""" 
        <style>
            .stApp {{
                background: url('{background_image_url}') no-repeat center center fixed;
                background-size: cover;
                transition: background 0.5s ease;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #f3f3f3;
                font-family: 'Roboto', sans-serif;
            }}
            .stButton>button {{
                color: #4e4376;
                background-color: #f3f3f3;
                transition: all 0.3s ease-in-out;
            }}
            .stButton>button:hover {{
                color: #f3f3f3;
                background-color: #2b5876;
            }}
            .stTextInput>div>div>input {{
                color: #4e4376;
                background-color: #f3f3f3;
            }}
        </style>
        """, unsafe_allow_html=True)

# Example usage with a background image URL
background_image_url = "https://images.fineartamerica.com/images-medium-large-5/singapore-skyline-panoramic-view--copyright-kengoh8888.jpg"
setting_bg(background_image_url)

st.markdown("""<div style='border:5px solid black; background-color:yellow; padding:10px;'> 
            <h1 style='text-align:center; color:red;'>Singapore  Resale Flat Prices Predicting</h1> </div>""", 
            unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(None, ["Home","Menu"], 
                    icons=["home","Menu"],
                    default_index=0,
                    orientation="vertical",  # Set orientation to vertical
                    styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "0px", "--hover-color": "#AB63FA", "transition": "color 0.3s ease, background-color 0.3s ease"},
                            "icon": {"font-size": "25px"},
                            "container" : {"max-width": "6000px", "padding": "10px", "border-radius": "5px","border": "5px solid black"},
                            "nav-link-selected": {"background-color": "red", "color": "white"}} )

if selected == "Home":
    st.markdown("## :red[**Overview :**] :white[Predicting resale flat prices in Singapore is a complex yet crucial task influenced by various factors such as location, size, age of the property, amenities, economic conditions, and government policies. With the Singapore housing market being dynamic and responsive to both internal and external stimuli, accurate prediction models play a vital role for stakeholders including buyers, sellers, policymakers, and real estate professionals. Leveraging advanced analytics, machine learning techniques, and historical data, researchers and analysts strive to develop models that offer insights into future price trends, aiding decision-making processes in the real estate market. By understanding the intricate interplay of variables affecting resale flat prices, these predictive models contribute to better-informed investments, policy formulation, and overall market stability]")

if selected  == "Menu":
        # Define unique values for select boxes
        flat_model_options = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
                        'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM',
                        'IMPROVED-MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT',
                        'ADJOINED FLAT', 'PREMIUM MAISONETTE', 'MODEL A2', 'DBSS', 'TYPE S1',
                        'TYPE S2', 'PREMIUM APARTMENT LOFT', '3GEN']
        flat_type_options = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
        town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
                        'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                        'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
                        'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
                        'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
        storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21',
                                '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30', '31 TO 33', '40 TO 42',
                                '37 TO 39', '34 TO 36', '46 TO 48', '43 TO 45', '49 TO 51']
       
        st.sidebar.markdown('<span style="color:red;font-size:30px;">Flat Details</span>', unsafe_allow_html=True)       
        town = st.sidebar.selectbox("Town", options=town_options)
        flat_type = st.sidebar.selectbox("Flat Type", options=flat_type_options)
        flat_model = st.sidebar.selectbox("Flat Model", options=flat_model_options)
        storey_range = st.sidebar.selectbox("Storey Range", options=storey_range_options)
        floor_area_sqm = st.sidebar.number_input("Floor Area (sqm)", min_value=0.0, max_value=500.0, value=100.0)
        current_remaining_lease = st.sidebar.number_input("Current Remaining Lease", min_value=0.0, max_value=99.0, value=20.0)
        year = 2024
        lease_commence_date = current_remaining_lease + year - 99
        years_holding = 99 - current_remaining_lease

        if st.sidebar.button("Predict Resale Price"):
                input_data = pd.DataFrame({
                        'town': [town],
                        'flat_type': [flat_type],
                        'flat_model': [flat_model],
                        'storey_range': [storey_range],
                        'floor_area_sqm': [floor_area_sqm],
                        'current_remaining_lease': [current_remaining_lease],
                        'lease_commence_date': [lease_commence_date],
                        'years_holding': [years_holding],
                        'remaining_lease': [current_remaining_lease],
                        'year': [year]})   
                
                
                import pickle

                with open(r'model_DTR.pkl', 'rb') as file:
                        model_dtr = pickle.load(file)
                with open(r'scaler_DTR.pkl', 'rb') as file:
                        scaler_dtr = pickle.load(file)
                with open(r'ohe.pkl', 'rb') as file:
                        ohe = pickle.load(file)
                with open(r'ohe2.pkl', 'rb') as file:
                        ohe2 = pickle.load(file)
                with open(r'ohe3.pkl', 'rb') as file:
                        ohe3 = pickle.load(file)
                with open(r'ohe4.pkl', 'rb') as file:
                        ohe4 = pickle.load(file)
                        

                #([['year','floor_area_sqm','lease_commence_date','years_holding','current_remaining_lease']].values, X1, X2, X3, X4)
                new_sample = np.array([[year,floor_area_sqm ,lease_commence_date,years_holding,current_remaining_lease,town,flat_type,flat_model,storey_range]])
                new_sample_town = ohe.transform(new_sample[:, [5]]).toarray()
                new_sample_flat_type = ohe2.transform(new_sample[:, [6]]).toarray()
                new_sample_flat_model = ohe3.transform(new_sample[:, [7]]).toarray()
                new_sample_storey_range = ohe4.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4]],new_sample_town,new_sample_flat_type,new_sample_flat_model,new_sample_storey_range ), axis=1)
                new_sample1 = scaler_dtr.transform(new_sample)
                new_pred = model_dtr.predict(new_sample1)[0]
                #st.write('## :green[Predicted Resale Price:] ',new_pred)
                st.markdown(f'<div style="background-color: white; border: 2px solid black; border-radius: 5px; padding: 10px; font-size: 32px; color: red;">Predicted Resale Price: <span style="color: darkgreen;">{new_pred}</span></div>', unsafe_allow_html=True)
