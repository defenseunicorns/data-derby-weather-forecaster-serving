import streamlit as st
import datetime
from dataclasses import dataclass
from weather.data import get_inputs_patch, get_labels_patch
from weather.model import WeatherModel
from visualize import show_inputs, show_outputs
from launch import Launch

@dataclass
class Model:
    name: str
    model: WeatherModel

@st.cache_resource
def load_models():
    models = {
        "26_100_epochs_tropics": Model(name="2/6 100 Epochs, Tropics", model=WeatherModel.from_pretrained("models/26_100_epochs_tropics")),
    }
    
    return models


def predict() -> None:
    point = (-80.607, 28.392)  # (longitude, latitude)
    patch_size = 128
    time = datetime.datetime.strptime(st.session_state.t, "%H:%M").time()
    dt = datetime.datetime.combine(st.session_state.d, time)
    st.session_state.i = get_inputs_patch(
        dt,
        point,
        patch_size,
    )

    st.session_state.predictions = models[st.session_state.model_name].model.predict(st.session_state.i)
    st.session_state.labels = get_labels_patch(dt, point, patch_size)


def reset():
    if st.session_state.input_type == "customtime":
        st.session_state.d = datetime.date(2022, 9, 30)
        st.session_state.t = "18:00"
    else:
        st.session_state.d = launches[st.session_state.input_type].date
        st.session_state.t = "18:00"
        predict()


if "input_type" not in st.session_state:
    st.session_state.input_type = "customtime"


launches = {
    "crew2demo": Launch(name="Crew 2 Demo", date=datetime.date(2020, 9, 30)),
    "starlink12": Launch(name="Falcon 9 Starlink-12", date=datetime.date(2020, 5, 10)),
}


def format_launch(id):
    return "Custom Time" if id == "customtime" else launches[id].name

def format_model(id):
    print(models)
    return models[id].name


models = load_models()
st.session_state.model_name = "26_100_epochs_tropics"

print(list(models.keys()))
st.sidebar.selectbox("Model", key="model_name", options=list(models.keys()), format_func=format_model)

st.sidebar.selectbox(
    "Launch to Predict",
    key="input_type",
    options=["customtime"] + list(launches.keys()),
    format_func=format_launch,
    on_change=reset,
)


st.write("# Demo")

if "i" in st.session_state:
    if st.checkbox("Input Data"):
        st.plotly_chart(show_inputs(st.session_state.i))

if "d" not in st.session_state:
    st.session_state.d = datetime.date(2022, 9, 30)

if "t" not in st.session_state:
    st.session_state.t = "18:00"


if st.session_state.input_type == "customtime":
    col1, col2 = st.columns(2)
    with col1:
        st.date_input("Forecast Date", key="d")
    with col2:
        st.text_input("Forecast Time (UTC)", key="t")

    st.button("Predict Weather", on_click=predict)


st.write(
    """
## Predictions
         """
)
if "predictions" in st.session_state:
    st.plotly_chart(show_outputs(st.session_state.predictions))

st.write(
    """
## Actual
"""
)

if "labels" in st.session_state:
    st.plotly_chart(show_outputs(st.session_state.labels))
