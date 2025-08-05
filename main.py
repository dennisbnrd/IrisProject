import streamlit as st
import numpy as np
import pandas as pd
import joblib

#page config
st.set_page_config(layout="centered", page_title="Iris Classifiaction", page_icon=":tulip:")

model = joblib.load('model.joblib')

def get_predict(data:pd.DataFrame, model):
    prediction = model.predict(data)
    predict_proba = model.predict_proba(data)
    map_label = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    prefiction_label = map(lambda x: map_label[x], list(prediction))

    return {
        "prediction": prediction,
        "predict_proba": predict_proba,
        "prefiction_label": list(prefiction_label)
    }

st.title('❀Iris Classifiaction❀')
st.write('Get the best result of iris classification with this app. Try it!')

sepal_info, petal_info = st.columns(2, gap="medium")

#sepal input
sepal_info.subheader("Sepal Information")
sepal_length = sepal_info.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.0)
sepal_width = sepal_info.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.0)

#petal input
petal_info.subheader("Petal Information")
petal_length = petal_info.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.0)
petal_width = petal_info.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.0)

predict = st.button("Predict", use_container_width=True)

if predict:
    df = pd.DataFrame({'sepal length (cm)': [sepal_length], 'sepal width (cm)': [sepal_width], 'petal length (cm)': [petal_length], 'petal width (cm)': [petal_width]}
                      )
    st.write(df)

    #prediction
    result = get_predict(df, model)

    label = result["prefiction_label"][0]
    prediction = result["prediction"][0]
    proba = result["predict_proba"][0][prediction]

    st.write(f"Your Iris Species is {proba:.0%} {label}")
 
    
