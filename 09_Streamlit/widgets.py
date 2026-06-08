import streamlit as st
import numpy as np
import pandas as pd

st.title("Streamlit Text Input")

name = st.text_input("Enter your name:")

age = st.slider("Select your age:", 0, 100, 25)

st.write(f"Your age is {age}.")


if name:
    st.write(f"Hello, {name}")