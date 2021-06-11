import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from jcopml.utils import save_model, load_model
from sklearn.model_selection import train_test_split
# Import Model Library
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Import HyperParameter Tuning Library
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Import model evaluation Library
from jcopml.plot import plot_classification_report, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import pickle

def satisfaction_airlines(file_name):
    data = pd.read_csv(file_name)
    data.dropna(inplace=True)
    data['satisfaction'] = data['satisfaction'].astype('category').cat.codes
    nums = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for var in nums:
        data[var]= (data[var]+1).apply(np.log)
        data[var]= MinMaxScaler().fit_transform(data[var].values.reshape(len(data), 1))
    nonordinal = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for cat in nonordinal:
        oneshoot = pd.get_dummies(data[cat], prefix=cat)
        data = data.join(oneshoot)
    data.drop(columns=nonordinal, inplace=True)
    ordinal = ['Seat comfort',
               'Departure/Arrival time convenient',
               'Food and drink',
               'Gate location',
               'Inflight wifi service',
               'Inflight entertainment',
               'Online support',
               'Ease of Online booking',
               'On-board service',
               'Leg room service',
               'Baggage handling',
               'Checkin service',
               'Cleanliness',
               'Online boarding',]
    for ords in ordinal:
        data[ords] = data[ords].replace(0, data[ords].mode()[0])
    return data  

data = satisfaction_airlines('Invistico_Airline.csv')
X = data.drop(columns=['satisfaction', 'Arrival Delay in Minutes'])
y = data['satisfaction'] # target / label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


def main():
    st.title("Satisfactory Airlines Prediction")
    st.markdown('This is the prediction of Airlines Passenger Satisfaction based on Machine Learning Model that we have developed')


    st.subheader('Entry your parameter here')
    age = st.slider('Passengers Age', min_value=0, max_value=80)
    fd = st.number_input('Flight Distance')
    seat = st.select_slider('Seat Comfort', [1,2,3,4,5])
    time_conv = st.select_slider('Departure/Arrival Time Convenient', [1,2,3,4,5])
    fnb = st.select_slider('Food and Drink', [1,2,3,4,5])
    gate = st.select_slider('Gate Location', [1,2,3,4,5])
    wifi = st.select_slider('Inflight Wifi Service', [1,2,3,4,5])
    ent = st.select_slider('Inflight Entertainment', [1,2,3,4,5])
    support = st.select_slider('Online Support', [1,2,3,4,5])
    booking = st.select_slider('Ease of Online Booking', [1,2,3,4,5])
    onboard = st.select_slider('On-board Service', [1,2,3,4,5])
    legroom = st.select_slider('Leg room Service', [1,2,3,4,5])
    baggage = st.select_slider('Baggage Handling', [1,2,3,4,5])
    checkin = st.select_slider('Checkin Service', [1,2,3,4,5])
    clean = st.select_slider('Cleanliness', [1,2,3,4,5])
    boarding = st.select_slider('Online Boarding', [1,2,3,4,5])
    delay = st.number_input('Departure Delay in Minutes')
    gender = st.selectbox('Gender', options=['Male','Female'])
    customer_type = st.selectbox('Customer Type', options=['Loyal','Disloyal'])
    travel_type = st.selectbox('Type of Travel', options=['Business Travel','Personal Travel'])
    travel_class = st.selectbox('Class', ['Eco','Eco Plus','Business'])
    predict = st.button('Predict')

    if gender=='Male':
        gender_male = 1
        gender_female = 0
    else:
        gender_male = 0
        gender_female = 1

    if customer_type=='Loyal':
        cust_loyal = 1
        cust_disloyal = 0
    else:
        cust_loyal = 0
        cust_disloyal = 1

    if travel_type=='Business Travel':
        bus_travel = 1
        pers_travel = 0
    else:
        bus_travel = 0
        pers_travel = 1

    if travel_class=='Eco':
        eco_class = 1
        ecop_class = 0
        bus_class = 0
    elif travel_class == 'Eco Plus':
        eco_class = 0
        ecop_class = 1
        bus_class = 0
    else:
        eco_class = 0
        ecop_class = 0
        bus_class = 1

    new = {
        'Age':age,
        'Flight Distance':fd,
        'Seat comfort':seat,
        'Departure/Arrival time convenient':time_conv,
        'Food and drink':fnb,
        'Gate location':gate,
        'Inflight wifi service':wifi,
        'Inflight entertainment':ent,
        'Online support':support,
        'Ease of Online booking':booking,
        'On-board service':onboard,
        'Leg room service':legroom,
        'Baggage handling':baggage,
        'Checkin service':checkin,
        'Cleanliness':clean,
        'Online boarding':boarding,
        'Departure Delay in Minutes':delay,
        'Gender_Female':gender_female,
        'Gender_Male':gender_male,
        'Customer Type_Loyal Customer':cust_loyal,
        'Customer Type_disloyal Customer':cust_disloyal,
        'Type of Travel_Business travel':bus_travel,
        'Type of Travel_Personal Travel':pers_travel,
        'Class_Business':bus_class,
        'Class_Eco':eco_class,
        'Class_Eco Plus':ecop_class
    }
    if predict:
        data_new = pd.DataFrame(data=new, index=[0])
        nums = ['Flight Distance', 'Departure Delay in Minutes']
        for var in nums:
            data_new[var]= (data_new[var]+1).apply(np.log)
            data_new[var]= MinMaxScaler().fit_transform(data_new[var].values.reshape(len(data_new), 1))
        model = XGBClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(data_new)
        percent_disatisfied = proba[0][0]
        percent_satisfied = proba[0][1]

        if percent_satisfied >= percent_disatisfied:
            st.write('Kemungkinan penumpang puas terhadap layanan adalah', percent_satisfied*100, '%')
        else:
            st.write('Kemungkinan penumpang tidak puas terhadap layanan adalah', percent_disatisfied*100, '%')
        
    




if __name__ == "__main__":
    main()