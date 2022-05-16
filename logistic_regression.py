# %%
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data yuklenir
print(st.title("Titanik faciəsindən sağ çıxa bilərdinizmi?"))

@st.cache
def model():
    # Data oxunur
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

    if p_class == "Birinci sinif":
        p_class = 1
    elif p_class == "İkinci sinif":
        p_class = 2
    else:
        p_class = 3
    # Dataframe hazırlayırıq
    input_data = {'Pclass': p_class, "Sex": gender, "Age": age, "SibSp": sibling}
    df = pd.DataFrame(data=input_data, index=[0])
    prediction = predict_data(model, df)
    predict_probability = model.predict_proba(df)
    if prediction[0] == 1:
        st.subheader('{}% ehtimalla sağ qalardınız.'.format(round(predict_probability[0][1] * 100, 3)))
    else:
        st.subheader('{}% ehtimalla ölərdiniz'.format(round(predict_probability[0][0] * 100, 3)))

# %%
#####Streamlit kitabxanasından istifadə edilir.
age = st.slider("Yaşınız", 1, 100, 1)
sibling = st.slider("Sizinlə birlikdə olan ailə üzvlərinizin sayı", 1, 10, 1)
gender = st.selectbox("Cins", options=["Kişi", "Qadın"])
p_class = st.selectbox("Sərnişin sinfi", options=['Birinci sinif', 'İkinci sinif', 'Üçüncü sinif'])
gender = 1 if gender == "Kişi" else 0
if st.button("Hesabla"):
    start(model)
