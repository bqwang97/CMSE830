import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

st.title("Explore the Forestfires Data")
df_forest = pd.read_csv("forestfires.csv") # load data
def month_to_quarter(month):
    if month in ['jan', 'feb', 'mar']:
        return 'Q1: Jan-Mar'
    elif month in ['apr', 'may', 'jun']:
        return 'Q2: Apr-Jun'
    elif month in ['jul', 'aug', 'sep']:
        return 'Q3: Jul-Sep'
    else:
        return 'Q4: Oct-Dec'
df_forest['quarter'] = df_forest['month'].apply(month_to_quarter)
df_forest['area'] = np.log1p(df_forest['area'])

st.header("Let's explore a sample of the dataset") # prints in web app
st.dataframe(df_forest.head()) # prints head in web app

st.header("Select X and Y Variables for the 'Forestfires' Dataset")
x_variable = st.selectbox("X Variable", df_forest.drop(columns=['X', 'Y']).columns)
y_variable = st.selectbox("Y Variable", df_forest.drop(columns=['X', 'Y']).columns)
selected_plots = st.multiselect("Select Plots to Display",
                                ["Scatter Plot","JointPlot","Heatmap","Histogram"],
                                default=["Scatter Plot"])

if "Scatter Plot" in selected_plots:
    st.subheader("Scatter Plot")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_forest, x=x_variable, y=y_variable)
    plt.title(f"Scatter plot between {x_variable} and {y_variable}")
    st.pyplot(plt)

if "JointPlot" in selected_plots:
    st.subheader("Jointplot")
    plt.figure(figsize=(8, 6))
    sns.jointplot(data=df_forest, x=x_variable, y=y_variable, kind="reg", color="#eccd13")
    #plt.title(f"Jointplot of {x_variable} vs {y_variable}")
    st.pyplot(plt)

if "Heatmap" in selected_plots:
    st.subheader("Heatmap")
    plt.figure(figsize=(10, 7))
    df_forestf1 = df_forest.drop(['X','Y','month','day'],axis =1)
    sns.heatmap(df_forestf1.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    st.pyplot(plt)

if "Histogram" in selected_plots:
    st.subheader("Histogram with Normal Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(df_forest[x_variable], kde=True)
    st.pyplot(plt)


import matplotlib.pyplot as plt # only using matplotlib to generate axes, not create plots


st.header("Spatial Distribution of Fires within the Montesinho park") #write figure title

fig, ax = plt.subplots(figsize=(11, 9))
sns.scatterplot(df_forest, x='X', y='Y', hue='area', size='area',sizes=(50,500))
st.pyplot(fig)


st.header("Temperature vs. Burned Area in Forest Fires")
fig = px.scatter(df_forest, 
                 x="temp", 
                 y="area", 
                 color="quarter",
                 size="area",
                 hover_data=['wind', 'rain'],)
                 #title="Temperature vs. Burned Area in Forest Fires")
st.plotly_chart(fig)