import streamlit as st
import pandas as pd
import numpy as np

st.markdown("""
# Cape Canaveral Wx Forecasting

## Team

### 🚀 Rocketing 🌈 Rainbow 🛡️ Defenders

* Gerred Dillon, 🤖🦄
* Jon Perry, 🤖🦄


## Objective

SLD45 and Cape Canaveral is growing to support 200+ launches per year. Weather violations are the primary cause of launch cancellations. Better identification and forecasting of weather will enable better planning and more launches.

Current weather forecasting models are based on physical simulation models, requiring significant compute resources and time to run. This project will explore the use of machine learning to predict weather conditions at launch and recovery sites.

## Hypothesis

Using state-of-the-art transformer models will support the 45th Weather Squadron in predicting weather conditions at launch and recovery sites.

## Approach

In a real world setting, the team wanted to use a multi-modal model that incorporated GOES16/ERA5 satellite data, weather radar stations, and other real-time telemetry sources. However, for the purposes of this hackathon, the team focused on a single input source: GOES16/ERA5 satellite data as data was readily available for a large time span and did not require complex data processing or training.

We investigated available foundational models attempting to predict weather conditions. Three models were selected for further research:

* Google MetNet
* Google MetNet-2
* Microsoft ClimaX

### MetNet / MetNet-2

MetNet was a first point of investigation and the CNN we wrote was based on MetNet. MetNet-2's model emulates physics based on Quasi-Geostrophic Theory, and approximates large-scale weather and storm onsets earlier than HREF, as well as the starting location of the storm. Most importantly, it is able to perform this task at 1²km resolution scales, in 2 minute increments, up to 12 hours in advance with a prediction time of 1s as compared to physics ensembling which can take upwards of 1 hour between observation and evaluation.

Unfortunately, MetNet-2 is closed source. While there is work going on to create open source, transformer-based models, they are early in development or still in training, and not yet available for use.


### Microsoft ClimaX

Released in February, ClimaX is a transformer-based model that uses a combination of satellite imagery and numerical weather prediction data to predict weather conditions. ClimaX is open source and available on GitHub. The model is able to produce weather data across climate, temporal, and spatial dimensions and can be fine tuned for anything from regional nowcasting to global seasional forecasting.

Foundational transformer models are ideal for this task, as they can be fine tuned for more specific tasks, such as nowcasting tropical conditions for the regions around Cape Canaveral.

Additionally, ClimaX is open source, with published weights. However, for the purposes of a hackathon, usage information on the model is still early.

### Custom CNN

We opted to go with a MetNet-2 style CNN, with a single convolutional layer for the purposes of the hackathon. It was quick to train, produced representative results, and enabled local inferencing. The most significant amount of time in an inference is spent in fetching map tiles, which ultimately may be alleviated by on-premise tile servers.

## Data

Initially, we chose a representative set of weather tiles based on a random sampling of 100 days across the past 7 years, over an even distribution of elevations and coordinates for an area covering the Americas. With this test, we were able to produce a training model with labeled tiles for times around predictions spanning -4, -2, 0, +2, and +6 hours.

Later, we re-trained the model based on a large tropical zone around Canaveral, covering both an inference area and an additional context area. Additionally, we adjusted the elevation map to be representative of the relatively lower elevations in this area (as compared to the greater Americas), and increased the sample of days to 1460 days.
""")

col1, col2 = st.columns(2)

america_points = [(-140.0, 60.0), (-140.0, -60.0), (-10.0, -60.0), (-10.0, 60.0)]
orig_df = pd.DataFrame(
    america_points,
    columns=['lon', 'lat'])
with col1:
    st.map(orig_df, zoom=1)     


tropics_points = [(36, -90), (18, -90), (18, -70), (36, -70)]
new_df = pd.DataFrame(
    tropics_points,
    columns=['lat', 'lon'])

with col2:
    st.map(new_df, zoom=2)
     
st.markdown("""
## Training

Data sets were then put through a CNN with a single convolution layer that is passed into a fully connected layer. Data was initially trained on 5 epochs, followed by 100 epochs for each data set.

## Validation

We did not have time to create a loss function beyond SmoothL1Loss. Future iterations, instead of relying on the output pixels of each vector to calculate accuracy and loss, would take a more semantic approach to storm intensity, start point, and other variables.

Additionally, the use of TorchViz and Weights and Biases would be useful additional evaluation tools. For the purposes of a Hackathon, having these tools ready in advance would enable their use.

## Conclusion / Demo / Lessons Learned

We believe this shows the potential for using Foundational Models with fine tuning for supporting the 45th Weather Squadron in better predicting weather violations. For future hackathons and work, we believe the following will be valuable for AI/ML at SLD45 and Cape Canaveral:

* More data, and more data types will enhance the ability to train and fine tune models
* Data simulation endpoints for more sensitive or time-series data
* Open source Foundational Models with LoRA (Low Rank Adaptation) for inexpensive, quick fine-tuning of foundational models
    * Shift to Transformer based models
* Small (1-13B) parameter Foundational Models trained exclusively on Mission Data
* Integration of small and large LLMs / other Foundational Models with small, specific models and fine tuned models
* Use of agents and LLMs for creating LLM goal-seeking task systems
* Tooling for evaluation of model performance
* Multi-modal model usage with both LLMs, anomaly detection, and sensor reading will have a significant impact on achieving SLD45's goals

## Other Notes

* We originally started with working with maintenance log data, and created a small model for determining maintenance time based on part ID and other factors. However, this model had only a few examples for each class, and so accuracy was hard to determine.
* Tile fetching is SLOW. The inference runs very quickly, most of the time is spent in fetching data. Local tile servers can help with this.
""")