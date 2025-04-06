import streamlit as st
import requests

st.set_page_config(page_title="Predição de Arremesso", layout="centered")

st.title("🏀 Predição de Arremesso do projeto Kobe")
st.markdown("Preencha os dados da jogada para prever se foi cesta:")

with st.form(key='predict_form'):
    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("Latitude", value=34.0443, format="%.4f")
        minutes_remaining = st.slider("Minutos Restantes", 0, 12, 10)
        playoffs = st.selectbox("É Playoff?", ["Não", "Sim"])

    with col2:
        lon = st.number_input("Longitude", value=-118.4268, format="%.4f")
        period = st.selectbox("Período do Jogo", [1, 2, 3, 4])
        shot_distance = st.slider("Distância do Arremesso (pés)", 0, 30, 18)

    submit_button = st.form_submit_button(label='🔮 Prever Resultado')

if submit_button:
    playoffs_val = 1 if playoffs == "Sim" else 0

    payload = {
        "dataframe_split": {
            "columns": ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"],
            "data": [[lat, lon, minutes_remaining, period, playoffs_val, shot_distance]]
        }
    }

    st.subheader("📦 Dados enviados ao modelo:")
    st.json(payload)

    try:
        response = requests.post("http://localhost:5001/invocations", json=payload)
        result = response.json()
        prediction = result.get("predictions", [None])[0]

        if prediction == 1.0:
            st.balloons()
            st.success("🏀 **Acertou a cesta!🔥")
        elif prediction == 0.0:
            st.error("❌ **Errou a cesta.**😓")
        else:
            st.warning(f"🔍 Resultado inesperado da predição: `{result}`")

    except Exception as e:
        st.error(f"❌ Erro ao consultar o modelo: {e}")