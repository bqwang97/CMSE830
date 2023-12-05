import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
st.image("https://www.greenpeace.org/static/planet4-international-stateless/2022/09/fbc851c4-gp1szphr_.jpg", width=700)
####################################################################################################################################################################
tab1, tab2 , tab3 , tab4 ,tab5 , tab6 , tab7= st.tabs(["Forest Fires Dataset", "Interactive Visualization","Regression Models","Neural Network Visualization","Neural Network Regression","",""])
####################################################################################################################################################################
##Forest Fires Dataset 
with tab1:                                                        
    st.title("Exploring the Forest Fires Dataset")
    ##st.subheader("Background")
    st.markdown('<p class="font_text">Considering the global warming, the increasing forest fires are more and more serious. </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">The primary goal of analysis of dataset "Forestfires" is to understand the interplay of various meteorological and spatial factors that influence forest fires occurrence and magnitude. By doing so, we aim to answer the questions below: What are the most influential determinants that lead to forest fires, and how could we predict future outbreaks and spread of these fires? If we can solve these problems, we can take preventive measures to minimize the air pollution and surrounding damage caused by forest fires.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> We choose the "Forestfires" dataset from UCI. This datasets provides a comprehensive view of both meteorological and spatial factors within the Montesinho park map, allowing for a detailed analysis of how these elements correlate with the extent of forest fires. The forestfires dataset originates from the Montesinho natural park in the Tra ́s-os-Montes northeast region of Portugal. The dataset was collected from January 2000 to December 2003 and it was built using two sources. There is no issue of missingness in the dataset. There are 516 rows and 13 columns in the dataset.</p>', unsafe_allow_html=True)
    
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
    df_forest['Logarea'] = np.log1p(df_forest['area'])
    
    col1,col2=st.columns(2,gap='small')
    forest_stat = col1.checkbox('Show statistical properties of Forest Fires dataset')
    if forest_stat==True:
        st.table(df_forest.describe())
        st.markdown('<p class="font_subtext">Table 1: Statistical properties of Forestfires attributes.</p>', unsafe_allow_html=True)

    forest_show = col2.checkbox('Show ForestFires Data')
    if forest_show==True:
        st.table(df_forest.head(20))
        st.markdown('<p class="font_subtext">Table 2: Forestfires dataset.</p>', unsafe_allow_html=True)

    st.sidebar.markdown('<p class="font_text">Dataset Description:</p>', unsafe_allow_html=True)

    df_columns = df_forest.columns
    selected_options = st.sidebar.multiselect("**Let's see the description of different columns present in the dataset. Select column names to see their brief description**", df_columns)
    description = {"X": "x-axis coordinate (from 1 to 9) within the park.", 
                   "Y": "y-axis coordinate (from 1 to 9) within the park.", 
                   "month": "Month of the year (January to December)", 
                   "day": "Day of the week (Monday to Sunday)",
                   "FFMC": "Fine Fuel Moisture Code denotes moisture content surface litter and influences ignition and fire spread. A high FFMC value suggests that the fine fuels are dry and conditions are suitable for the easy spread of fire.", 
                   "DMC": "Duff Moisture Code represents moisture content of shallow organic layers which affect fire intensity.", 
                   "DC": "Drought Code is an index of the moisture content of deep, compact organic layers. High DC values indicates that the deep organic layers are dry and there is a higher risk of more intense fires ", 
                   "ISI": "Initial Spread Index correlates with fire velocity spread. High ISI value occurs during conditions of high wind and low fine fuel moisture content, suggesting rapid fire spread.", 
                   "temp": "Outside temperature (in Celsius )",
                   "RH": "Outside relative humidity (in %)",
                   "wind": "Outside wind speed (in km/h)",
                   "rain": "Outside rain (in mm/m )",
                   "area": "Total burned area (in ha)",
                   "quarter": " I added quarter column by dividing the month into four groups: Jan to Mar, Apr to Jun, July to Sep and Oct to Dec. In this way, the distribution of the fires could be easier be visualized",
                   "Logarea": "To reduce skewness and improve symmetry, the logarithm function y = ln(x + 1) was applied to the area attribute"
                   }
    for option in selected_options:
            st.sidebar.markdown(f"**Description of {option}:** {description[option]}")
##############################################################################################################################################
with tab2:
    st.markdown('<p class="font_header">Interactive Visualization: </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">Several visualization are developed to study possible existing trend between different features of the dataset. Moreover, some of the figures are based on the target variable of the dataset which is the burned area and its spatial distribution. </p>', unsafe_allow_html=True)

    st.sidebar.markdown('<p class="font_text">Fig. 4: Matrix plot configuration:</p>', unsafe_allow_html=True)
    col1,col2=st.columns(2,gap='small')

    pairplot_options_x = col1.multiselect('Select features for x-axis of pairplot:',df_forest.drop(columns=['month','day','X', 'Y']).columns,default = "temp")
    pairplot_options_y = col2.multiselect('Select features for y-axis of pairplot:',df_forest.drop(columns=['month','day','X', 'Y']).columns,default = "Logarea")
    pairplot_hue = st.sidebar.select_slider('Select hue for matrixplot:',options=['quarter', 'month'])
    #hue = pairplot_hue if pairplot_hue != 'None' else None
    fig1 = sns.pairplot(data=df_forest,x_vars=pairplot_options_x,y_vars=pairplot_options_y, hue=pairplot_hue, palette='hsv')
    
    df_forestf1 = df_forest.drop(['X','Y','month','day'],axis =1)
    
    c=alt.Chart(df_forest).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=pairplot_hue,
        tooltip=['X', 'Y']
    ).properties(
        width=280,
        height=280
    ).repeat(
        row=pairplot_options_y,
        column=pairplot_options_x
    ).interactive()

    st.altair_chart(c, use_container_width=True)
    
    tab8, tab9,tab10 = st.tabs(["Heatmap", "Histogram", "Contourplot"])
    with tab8:
        plt.figure(figsize=(8, 7))
        df_forestf1 = df_forest.drop(['X','Y','month','day'],axis =1)
        sns.heatmap(df_forestf1.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.markdown('<p class="font_subtext">Fig. 5: Feature Correlation Heatmap.</p>', unsafe_allow_html=True)
        ##plt.title("Feature Correlation Heatmap")
        st.pyplot(plt)
    with tab9:
        st.markdown('<p class="font_subtext">Fig. 5: Histplot Showing the difference for area and logarea distribution. </p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2,gap='small')
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(data=df_forest, x='area', ax=ax)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(data=df_forest, x='Logarea', ax=ax)
            st.pyplot(fig)
    with tab10:
        st.markdown('<p class="font_subtext">Fig. 5: Contour Plot Showing Influence on Burned Area. </p>', unsafe_allow_html=True)
        st.markdown('<p class="font_text"> From the previous visualization, the direct relationship between weather indicators and the extent of burned areas was not immediately clear. Hence, we decided to focus on pairs of weather features, visualizing them through 2D contour plots. This approach aims to provide a clearer perspective on their combined influence on forest fires. </p>', unsafe_allow_html=True)
        
        option1 = st.selectbox('Feature 1', ('FFMC','DMC','DC','ISI','temp','RH','wind','rain'),index =1)
        option2 = st.selectbox('Feature 2', ('FFMC','DMC','DC','ISI','temp','RH','wind','rain'),index =2)
        
        fig = px.density_contour(df_forest, x=option1, y= option2, z='area',histfunc="avg",
                                 labels={'area': 'Burned Area'},width=800, height=600)
        fig.update_traces(contours_coloring="fill", contours_showlabels = True,colorscale='Spectral')
        st.plotly_chart(fig)   
    ###################################################################################################
    st.header("Spatial Distribution of Fires within the Montesinho park") #write figure title
    st.markdown('<p class="font_text"> The spatial distribution of fires within the Montesinho Park is very important to investigate the fire trends. From the scatter plot we could see the location where the fire happens. </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> The visualizations effectively communicate the spatial distribution of forest fires throughout different times of the year, emphasizing the significance of the third quarter (July to September) in the frequency of fires. </p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2,gap='small')
    
    col1.subheader("3D Distribution of Fires within the Montesinho park vs Quarter") #write figure title
    fig=px.scatter_3d(df_forest, x='X', y='Y', z="Logarea",color="quarter")
    col1.plotly_chart(fig)
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(df_forest, x='X', y='Y', hue="Logarea", size="Logarea",sizes=(50,500))
    col2.subheader("2D Distribution of Burned Area")
    col2.pyplot(fig)
####################################################################################################################################################################
#Regression Models
with tab3:
    st.markdown('<p class="font_header">Linear Regression:</p>', unsafe_allow_html=True)
    Scaler = st.checkbox('Applying Scaler object for linear regression fitting')
    df_forest_scaler = df_forest.drop(['X','Y','month','day'], axis=1)
    col1 , col2 = st.columns(2,gap='small')
    Feature_Variable = col1.multiselect('Select feature(s) for linear regression:',
                                        ['FFMC','DMC','DC','ISI','temp','RH','wind','rain'], default = 'temp')
    X = df_forest_scaler[Feature_Variable]
    y = df_forest_scaler['Logarea']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    Index=np.linspace(0,y_test.size-1,y_test.size).astype(int)

    if Scaler:
        Scaler_Type = col2.selectbox('Select scaler object:',['Min-Max Scaler', 'Standard Scaler'],index = 1)
        if Scaler_Type == 'Min-Max Scaler':
            Scaler_Object = MinMaxScaler()
        elif Scaler_Type == 'Standard Scaler':
            Scaler_Object = StandardScaler()
        Scaler_Object.fit(X_train)    
        X_train_scaled = Scaler_Object.transform(X_train)
        X_test_scaled = Scaler_Object.transform(X_test)
    Linear_Regression_Object = LinearRegression()         
    if Scaler:
        Linear_Regression_Object.fit(X_train_scaled, y_train)
        lin_reg_predictions = Linear_Regression_Object.predict(X_test_scaled)
        lin_reg_mse = mean_squared_error(y_test, lin_reg_predictions)
        lin_reg_r2 = r2_score(y_test, lin_reg_predictions)
    else:
        Linear_Regression_Object.fit(X_train, y_train)
        lin_reg_predictions = Linear_Regression_Object.predict(X_test)
        lin_reg_mse = mean_squared_error(y_test, lin_reg_predictions)
        lin_reg_r2 = r2_score(y_test, lin_reg_predictions)
    st.write('For linear regression methods ', 'the accuracy score based on r2 ',np.round(lin_reg_r2),'.')
    st.write('For linear regression methods ', 'the Mean Squared Error is  ',np.round(lin_reg_mse),'.')
    Linear_Dataframe=pd.DataFrame(index=np.arange(len(y_test)), columns=np.arange(3))
    Linear_Dataframe.columns=['Index','Actual','Predict']
    Linear_Dataframe['Index'] = Index
    Linear_Dataframe['Actual'] = y_test.reset_index(drop=True)
    Linear_Dataframe['Predict'] = lin_reg_predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Linear_Dataframe['Index'], y=Linear_Dataframe['Actual'],marker_symbol='square',
                        mode='markers',
                        name='Actual'))
    fig.add_trace(go.Scatter(x=Linear_Dataframe['Index'], y=Linear_Dataframe['Predict'],marker_symbol='circle',
                        mode='markers',
                        name='Prediction'))

    st.plotly_chart(fig)
    ###################################################################################################
    st.markdown('<p class="font_header">Random Forest:</p>', unsafe_allow_html=True)
    col1 , col2, col3,col4= st.columns(4,gap='small')
    Feature_Variable = col1.multiselect('Select feature(s) for linear regression:',
                                        ['FFMC','DMC','DC','ISI','temp','RH','wind','rain'], default = 'temp')
    Estimator = col2.slider('Input a value for estimator',0,200)
    Random_State_rf = col3.slider('Input a value for random state', 0, 200, 40)
    Random_Forest_Object = RandomForestRegressor(n_estimators=Estimator, random_state=Random_State_rf)

    Scaler_Type = col4.selectbox('Select scaler object:',['Min-Max Scaler', 'Standard Scaler'],index = 1)
    if Scaler_Type == 'Min-Max Scaler':
        Scaler_Object = MinMaxScaler()
    elif Scaler_Type == 'Standard Scaler':
        Scaler_Object = StandardScaler()
    Scaler_Object.fit(X_train)    
    X_train_scaled = Scaler_Object.transform(X_train)
    X_test_scaled = Scaler_Object.transform(X_test)

    
    Random_Forest_Object.fit(X_train_scaled, y_train)
    rf_reg_predictions = Random_Forest_Object.predict(X_test_scaled)
    rf_reg_mse = mean_squared_error(y_test, rf_reg_predictions)
    rf_reg_r2 = r2_score(y_test, rf_reg_predictions)
    st.write('For random forest regression methods ', 'the accuracy score based on r2 ',np.round(rf_reg_r2),'.')
    st.write('For random forest regression methods ', 'the Mean Squared Error is  ',np.round(rf_reg_mse),'.')

    Index=np.linspace(0,y_test.size-1,y_test.size).astype(int)
    RF_Dataframe=pd.DataFrame(index=np.arange(len(y_test)), columns=np.arange(3))
    RF_Dataframe.columns=['Index','Actual','Predict']
    RF_Dataframe['Index'] = Index
    RF_Dataframe['Actual'] = y_test.reset_index(drop=True)
    RF_Dataframe['Predict'] = rf_reg_predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=RF_Dataframe['Index'], y=RF_Dataframe['Actual'],marker_symbol='square',
                        mode='markers',
                        name='Actual'))
    fig.add_trace(go.Scatter(x=RF_Dataframe['Index'], y=RF_Dataframe['Predict'],marker_symbol='circle',
                        mode='markers',
                        name='Prediction'))

    st.plotly_chart(fig)
####################################################################################################################################################################
#Reference
st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Cortez, P., & Morais, A. D. J. R. (2007). A data mining approach to predict forest fires using meteorological data. </p>', unsafe_allow_html=True)
