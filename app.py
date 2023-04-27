import streamlit as st
import datetime
import numpy as np
from plotly.graph_objects import Figure
import plotly.graph_objects as graph_objects
from plotly.subplots import make_subplots

from weather.model import WeatherModel
from weather.data import get_inputs_patch, get_labels_patch


@st.cache_resource
def load_model():
    print("hit")
    model_path = "model/"
    return WeatherModel.from_pretrained(model_path)


def show_outputs(patch: np.ndarray) -> Figure:
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(graph_objects.Image(z=render_gpm(patch[:, :, 0:1])), row=1, col=1)
    fig.add_trace(graph_objects.Image(z=render_gpm(patch[:, :, 1:2])), row=1, col=2)
    fig.update_layout(height=300, margin=dict(l=0, r=0, b=0, t=0))
    return fig


def render_gpm(patch: np.ndarray) -> np.ndarray:
    palette = [
        "000096",  # Navy blue
        "0064ff",  # Blue ribbon blue
        "00b4ff",  # Dodger blue
        "33db80",  # Shamrock green
        "9beb4a",  # Conifer green
        "ffeb00",  # Turbo yellow
        "ffb300",  # Selective yellow
        "ff6400",  # Blaze orange
        "eb1e00",  # Scarlet red
        "af0000",  # Bright red
    ]
    return render_palette(patch[:, :, 0], palette, max=20)


def render_palette(
    values: np.ndarray, palette: list[str], min: float = 0.0, max: float = 1.0
) -> np.ndarray:
    """Renders a NumPy array with shape (width, height, 1) as an image with a palette.

    Args:
        values: An uint8 array with shape (width, height, 1).
        palette: List of hex encoded colors.

    Returns: An uint8 array with shape (width, height, rgb) with colors from the palette.
    """
    # Create a color map from a hex color palette.
    xs = np.linspace(0, len(palette), 256)
    indices = np.arange(len(palette))

    red = np.interp(xs, indices, [int(c[0:2], 16) for c in palette])
    green = np.interp(xs, indices, [int(c[2:4], 16) for c in palette])
    blue = np.interp(xs, indices, [int(c[4:6], 16) for c in palette])
    color_map = np.array([red, green, blue]).astype(np.uint8).transpose()

    scaled_values = (values - min) / (max - min)
    color_indices = (scaled_values.clip(0, 1) * 255).astype(np.uint8)
    return np.take(color_map, color_indices, axis=0)


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

    st.session_state.predictions = model.predict(st.session_state.i)
    st.session_state.labels = get_labels_patch(dt, point, patch_size)

def show_inputs(patch: np.ndarray) -> Figure:
    fig = make_subplots(rows=2, cols=4)
    fig.add_trace(graph_objects.Image(z=render_gpm(patch[:, :, 0:1])), row=1, col=1)
    fig.add_trace(graph_objects.Image(z=render_gpm(patch[:, :, 1:2])), row=1, col=2)
    fig.add_trace(graph_objects.Image(z=render_gpm(patch[:, :, 2:3])), row=1, col=3)
    fig.add_trace(graph_objects.Image(z=render_goes16(patch[:, :, 3:19])), row=2, col=1)
    fig.add_trace(
        graph_objects.Image(z=render_goes16(patch[:, :, 19:35])), row=2, col=2
    )
    fig.add_trace(
        graph_objects.Image(z=render_goes16(patch[:, :, 35:51])), row=2, col=3
    )
    fig.add_trace(
        graph_objects.Image(z=render_elevation(patch[:, :, 51:52])), row=1, col=4
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=0))
    return fig

def render_goes16(patch: np.ndarray) -> np.ndarray:
    red = patch[:, :, 1]  # CMI_C02
    green = patch[:, :, 2]  # CMI_C03
    blue = patch[:, :, 0]  # CMI_C01
    rgb_patch = np.stack([red, green, blue], axis=-1)
    return render_rgb_images(rgb_patch, max=3000)

def render_rgb_images(
    values: np.ndarray, min: float = 0.0, max: float = 1.0
) -> np.ndarray:
    """Renders a numeric NumPy array with shape (width, height, rgb) as an image.

    Args:
        values: A float array with shape (width, height, rgb).
        min: Minimum value in the values.
        max: Maximum value in the values.

    Returns: An uint8 array with shape (width, height, rgb).
    """
    scaled_values = (values - min) / (max - min)
    rgb_values = scaled_values.clip(0, 1) * 255
    return rgb_values.astype(np.uint8)


def render_elevation(patch: np.ndarray) -> np.ndarray:
    palette = [
        "000000",  # Black
        "478fcd",  # Shakespeare blue
        "86c58e",  # De York green
        "afc35e",  # Celery green
        "8f7131",  # Pesto brown
        "b78d4f",  # Muddy waters brown
        "e2b8a6",  # Rose fog pink
        "ffffff",  # White
    ]
    return render_palette(patch[:, :, 0], palette, max=3000)



model = load_model()
input_type = st.sidebar.radio("custom time input", options=["custom time input", "scrub one", "scurb two", "hurricane ian"])


st.write("# Cape Canaveral Weather Forecast")

if "predictions" in st.session_state:
    if st.checkbox("Show Raw Data"):
        st.subheader("Input Images")
        st.plotly_chart(show_inputs(st.session_state.i))

if "d" not in st.session_state:
    st.session_state.d = datetime.date(2022, 9, 30)

if "t" not in st.session_state:
    st.session_state.t = "18:00"


match input_type:
    case "custom time input":
        col1, col2 = st.columns(2)
        with col1:
            st.date_input("Forecast Date", key="d")
        with col2:
            st.text_input("Forecast Time (UTC)", key="t")

        st.button("Predict Weather", on_click=predict)
    case "scrub one":
        st.session_state.d = datetime.date(2022, 9, 30)
        st.session_state.t = "18:00"
        predict()



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
