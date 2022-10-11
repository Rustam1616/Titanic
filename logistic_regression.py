# %%
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data yuklenir
print(st.title("How likely were you to survive the Titanic?"))

@st.cache
def model():
    # Data readed
    data = pd.read_csv("train.csv")

    le = LabelEncoder()

    data["Sex"] = le.fit_transform(data["Sex"])

    # %%
    data.drop(["Name", "Ticket", "Cabin", "PassengerId"], inplace=True, axis=1)
    data.dropna(inplace=True)
    x = data.drop(["Survived", "Parch", "Fare", "Embarked"], axis=1)
    y = data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


model = model()


def predict_data(model, user_input):
    result = model.predict(user_input)
    return result


def start(model):
    # Global variables defined
    global p_class
    global gender
    global age
    global sibling

    if p_class == "First class":
        p_class = 1
    elif p_class == "Second class":
        p_class = 2
    else:
        p_class = 3
    # Dataframe preparation
    input_data = {'Pclass': p_class, "Sex": gender, "Age": age, "SibSp": sibling}
    df = pd.DataFrame(data=input_data, index=[0])
    prediction = predict_data(model, df)
    predict_probability = model.predict_proba(df)
    if prediction[0] == 1:
        st.subheader('Would be survived with {}% probability.'.format(round(predict_probability[0][1] * 100, 3)))
        st.image("https://www.denofgeek.com/wp-content/uploads/2015/10/raise-main.jpg?resize=620%2C349")
    else:
        st.subheader('Would be died with {}% probability'.format(round(predict_probability[0][0] * 100, 3)))
        st.image("https://media.nationalgeographic.org/assets/photos/000/273/27302.jpg")

# %%
#####stramlit library was used.
age = st.slider("Your age", 1, 100, 1)
sibling = st.slider("Number of family members travelling with you", 1, 10, 1)
gender = st.selectbox("Sex", options=["Male", "Female"])
p_class = st.selectbox("Passenger class", options=['First class', 'Second class', 'Third class'])
gender = 1 if gender == "Male" else 0
if st.button("Check"):
    start(model)
