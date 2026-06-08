import streamlit as st 
import pandas as pd
import numpy as np

# Title of the application
st.title("Hello Streamlit")


## display a simplet text
st.write("This is a simplet text")


## create a simple dataframe
df = pd.DataFrame({
    'firs column': [1,2,3,4],
    'second column': [10, 20, 30, 40]
})

## Display the DataFrame
st.write("Here is the dataFrame")
st.write(df)


# create a line chate

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(chart_data)