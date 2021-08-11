import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRegressor


@st.cache()
def loader(
    data_path="./data/model_input.csv",
    model_path="./model/model.cbm",
    normalizer_path="./model/data_pipeline.pkl",
):
    model_input = pd.read_csv(data_path)
    with open(normalizer_path, "rb") as f:
        data_pipeline = pickle.load(f)

    model = CatBoostRegressor()
    model.load_model(model_path)
    return model_input, data_pipeline, model


model_input, data_pipeline, model = loader()
st.title("Calculadora de imóveis")

st.sidebar.title("Entre com as características do apartamento")
area = st.sidebar.number_input("Area", min_value=10, max_value=400, step=25, value=70)
bairro = st.sidebar.selectbox("Bairro", options=model_input["bairro"].unique())
garages = st.sidebar.slider("Garagens", min_value=0, max_value=5)
bathrooms = st.sidebar.slider("Banheiros", min_value=1, max_value=5)
rooms = st.sidebar.slider("Quartos", min_value=1, max_value=5)

novo_apto = pd.DataFrame(
    [[area, rooms, bathrooms, garages, bairro]],
    columns=["area", "rooms", "bathrooms", "garages", "bairro"],
)

novo_apto_normalizado = pd.DataFrame(
    data_pipeline.transform(novo_apto), columns=novo_apto.columns
)

# st.dataframe(novo_apto)
# st.dataframe(novo_apto_normalizado)
prediction = model.predict(novo_apto_normalizado)[0]

st.write(f"Preço previsto: R$ {prediction:,.2f}")
mensagem = st.text_input("Envie seu feedback")

with open("./report/mensagem_usuario.txt", "a") as f:
    print("Escrevendo mensagem")
    print(f"{mensagem}")
    f.write(mensagem)
    f.write("\n")
