# TODO: Import Reuired Libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# TODO: Build Model
df = pd.read_csv(filepath_or_buffer="./iris.csv")
# print(df)
target = "species"
features = df.columns.to_list()
features.remove(target)
# print(target)
# print(features)
X = df[features].values
y = df[target].values
robot = RandomForestClassifier().fit(X=X, y=y)
# X_test = X[[0], :]
# y_pred = robot.predict(X=X_test)
# print(X_test)
# print(y_pred)

# TODO: Build Web Application

st.title(body="Predict Iris Species")
st.header(body="Iris Data Set")
st.dataframe(data=df.head())
# st.write(features)
st.sidebar.header(body="User Input")
sepal_length = st.sidebar.number_input(
    label="sepal_length",
    value=df["sepal_length"].median(),
    step=0.1,
)

sepal_width = st.sidebar.number_input(
    label="sepal_width",
    value=df["sepal_width"].median(),
    step=0.1,
)

petal_length = st.sidebar.number_input(
    label="petal_length",
    value=df["petal_length"].median(),
    step=0.1,
)

petal_width = st.sidebar.number_input(
    label="petal_width",
    value=df["petal_width"].median(),
    step=0.1,
)
# st.write(sepal_length)
X_test = [[sepal_length, sepal_width, petal_length, petal_width]]
user_input = pd.DataFrame(data=X_test, columns=features)
st.header(body="User Input")
st.dataframe(data=user_input)
y_pred = robot.predict(X=X_test)
y_prob = robot.predict_proba(X=X_test)
# st.write(X_test)
st.header(body="Prediction")
st.write(y_pred[0])
# st.write(y_prob)
