import streamlit as st

st.markdown("""
# Cape Canaveral Wx Forecasting

## Objective

SLD45 and Cape Canaveral is growing to support 200+ launches per year. Weather violations is the primary cause of launch cancellations. Better identification and forecasting of weather will enable better planning and more launches.

Current weather forecasting models are based on physical simulation models, requiring significant compute resources and time to run. This project will explore the use of machine learning to predict weather conditions at launch and recovery sites.

## Hypothesis

Using state-of-the-art transformer models will support the 45th Weather Squadron in predicting weather conditions at launch and recovery sites.

## TODO:

* Talk about other models
    * MetNet / MetNet-2
    * Microsoft ClimaX
* Talk about data
    * Launch data
* Talk about model used
* Lessons Learned
* Lessons for SLD45 / Future Work
""")