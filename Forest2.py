import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
##################################################################################################################################################################
st.markdown(""" <style> .font_title {
font-size:50px ; font-family: 'times '; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:50px ; font-family: "Times New Roman"; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:35px ; font-family: "Times New Roman" ; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:28px ; font-family: "Times New Roman" ; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:22px ; font-family: "Times New Roman" ; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext {
font-size:18px ; font-family: "Times New Roman" ; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)
####################################################################################################################################################################
st.title("Exploring the Forest Fires Dataset")
col1, col2= st.columns([1,3])
col1.subheader("Background")
col1.markdown('<p class="font_text">Considering the global warming, the increasing forest fires are more and more serious. </p>', unsafe_allow_html=True)
col1.markdown('<p class="font_text">The primary goal of analysis of dataset "Forestfires" is to understand the interplay of various meteorological and spatial factors that influence forest fires occurrence and magnitude. By doing so, we aim to answer the questions below: What are the most influential determinants that lead to forest fires, and how could we predict future outbreaks and spread of these fires? If we can solve these problems, we can take preventive measures to minimize the air pollution and surrounding damage caused by forest fires.</p>', unsafe_allow_html=True)
col2.image("https://www.greenpeace.org/static/planet4-international-stateless/2022/09/fbc851c4-gp1szphr_.jpg", width=700)

col2.divider()
####################################################################################################################################################################
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
df_forest= df_forest[df_forest['area'] !=0]
df_forest['Logarea'] = np.log1p(df_forest['area'])


col1.subheader("Dataset Description ")
col1.markdown('<p class="font_text"> We choose the "Forestfires" dataset from UCI. This datasets provides a comprehensive view of both meteorological and spatial factors within the Montesinho park map, allowing for a detailed analysis of how these elements correlate with the extent of forest fires. The forestfires dataset originates from the Montesinho natural park and includes several parameters:* Spatial Coordinates: X and Y axis spatial coordinates within the park.* Time Factors: Month and day of the week* Fire weather Index System Parameters: Include FFMC(index of the moisture content of surface litter), DMC(index of the moisture content of organic layers), DC(index of the moisture content of deep, compact organic layers) and ISI(index of the expected rate of fire spread) indices.* Meteorological Data: Temperature (in Celsius), relative humidity (%), wind speed (km/h), and outside rain (mm/m^2).* Outcome Variable: Burned area of the forest (in ha). </p>', unsafe_allow_html=True)

col2.header("Let's explore the dataset")
col2.dataframe(df_forest.head()) # prints head in web app

col2.divider()
##############################################################################################################################################
col2.header("Select X and Y Variables for the 'Forestfires' Dataset")
col2.markdown('<p class="font_text"> Several visualizations are developed to study possible existing trend between different features of the dataset. </p>', unsafe_allow_html=True)

col2a, col2b = col2.columns([1,2])
x_variable = col2a.selectbox("X Variable", df_forest.drop(columns=['X', 'Y']).columns)
y_variable = col2a.selectbox("Y Variable", df_forest.drop(columns=['X', 'Y']).columns)
selected_plots = col2a.multiselect("Select Plots to Display",
                                ["Scatter Plot","JointPlot","Heatmap","Histogram"],
                                default=["Scatter Plot"])

if "Scatter Plot" in selected_plots:
    col2b.subheader("Scatter Plot")
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df_forest, x=x_variable, y=y_variable,color = 'red')
    ##plt.title(f"Scatter plot between {x_variable} and {y_variable}")
    col2b.pyplot(plt)

if "JointPlot" in selected_plots:
    col2b.subheader("Jointplot")
    plt.figure(figsize=(8, 6))
    sns.jointplot(data=df_forest, x=x_variable, y=y_variable, kind="reg", color="g")
    #plt.title(f"Jointplot of {x_variable} vs {y_variable}")
    col2b.pyplot(plt)

if "Heatmap" in selected_plots:
    col2b.subheader("Heatmap")
    plt.figure(figsize=(8, 7))
    df_forestf1 = df_forest.drop(['X','Y','month','day'],axis =1)
    sns.heatmap(df_forestf1.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    col2b.pyplot(plt)

if "Histogram" in selected_plots:
    col2b.subheader("Histogram with Normal Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(df_forest[x_variable], kde=True)
    col2b.pyplot(plt)

col2.divider()
###################################################################################################
col1.header("Spatial Distribution of Fires within the Montesinho park") #write figure title
col1.markdown('<p class="font_text"> The spatial distribution of fires within the Montesinho Park is very important to investigate the fire trends. From the scatter plot we could see the location where the fire happens. </p>', unsafe_allow_html=True)
col1.markdown('<p class="font_text"> The visualizations effectively communicate the spatial distribution of forest fires throughout different times of the year, emphasizing the significance of the third quarter (July to September) in the frequency of fires. </p>', unsafe_allow_html=True)

col2a, col2b = col2.columns(2)

col2a.subheader("3D Distribution of Fires within the Montesinho park vs Quarter") #write figure title
fig=px.scatter_3d(df_forest, x='X', y='Y', z="Logarea",color="quarter")
col2a.plotly_chart(fig)

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(df_forest, x='X', y='Y', hue="Logarea", size="Logarea",sizes=(50,500))
col2b.subheader("2D Distribution of Burned Area")
col2b.pyplot(fig)

col2.divider()
###################################################################################################
col2.header("Temperature vs. Burned Area in Forest Fires")
col2.markdown('<p class="font_text"> When we visualize the relationship between the burned area and temperature, it is evident that as the temperature rises, the area affected by fires tends to increase. Additionally, by segmenting the data into different quarters, we can gain insights into how fire occurrences vary across specific months. </p>', unsafe_allow_html=True)
df_forest['quarter'] = df_forest['quarter'].astype('category')
color_map1 = {
    'Q1: Jan-Mar': "red",     
    'Q2: Apr-Jun': "blue",    
    'Q3: Jul-Sep': "orange",   
    'Q4: Oct-Dec': "brown" }
fig = px.scatter(df_forest, 
                 x="temp", 
                 y="Logarea", 
                 color="quarter",
                 size="Logarea",
                 hover_data=['wind', 'rain'],
                 color_discrete_map= color_map1)
                 #title="Temperature vs. Burned Area in Forest Fires")
col2.plotly_chart(fig)

col2.divider()
###################################################################################################

col2.header("Contour Plot Showing Influence on Burned Area")
col2.markdown('<p class="font_text"> From the previous visualization, the direct relationship between weather indicators and the extent of burned areas was not immediately clear. Hence, we decided to focus on pairs of weather features, visualizing them through 2D contour plots. This approach aims to provide a clearer perspective on their combined influence on forest fires. </p>', unsafe_allow_html=True)

col2a,col2b = col2.columns([1,3])
option1 = col2a.selectbox('Feature 1', ('FFMC','DMC','DC','ISI','temp','RH','wind','rain'),index =1)
option2 = col2a.selectbox('Feature 2', ('FFMC','DMC','DC','ISI','temp','RH','wind','rain'),index =2)

fig = px.density_contour(df_forest, x=option1, y= option2, z='area',histfunc="avg",
                         labels={'area': 'Burned Area'},width=800, height=600)
fig.update_traces(contours_coloring="fill", contours_showlabels = True,colorscale='Spectral')
col2b.plotly_chart(fig)
####################################################################################################################################################################
#Reference
st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Cortez, P., & Morais, A. D. J. R. (2007). A data mining approach to predict forest fires using meteorological data. </p>', unsafe_allow_html=True)
