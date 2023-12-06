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
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ExpSineSquared,DotProduct
from sklearn.multioutput import MultiOutputRegressor


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
tab1, tab2 , tab3 , tab4 ,tab5 , tab6 , tab7= st.tabs(["Forest Fires Dataset", "Interactive Visualization","Linear Regression","Random Forest","Neural Network Regression","Support Vector Machines",""])
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
#Linear Regression
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

################################################################################################################################
# Random forest
with tab4:    
    st.markdown('<p class="font_header">Random Forest:</p>', unsafe_allow_html=True)
    col1 , col2, col3,col4= st.columns(4,gap='small')
    Feature_Variable2 = col1.multiselect('Select feature(s) for random forest model:',
                                        ['FFMC','DMC','DC','ISI','temp','RH','wind','rain'], default = 'temp')
    X_rf = df_forest_scaler[Feature_Variable2]
    y_rf = df_forest_scaler['Logarea']
    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2)
    
    Estimator = col2.slider('Input a value for estimator',10,200)
    Random_State_rf = col3.slider('Input a value for random state', 0, 200, 40)
    Random_Forest_Object = RandomForestRegressor(n_estimators=Estimator, random_state=Random_State_rf)

    Scaler_Type_rf = col4.selectbox('Select RF scaler object:',['Min-Max Scaler', 'Standard Scaler'],index = 1)
    if Scaler_Type_rf == 'Min-Max Scaler':
        Scaler_Object_rf = MinMaxScaler()
    elif Scaler_Type_rf == 'Standard Scaler':
        Scaler_Object_rf = StandardScaler()
    Scaler_Object_rf.fit(X_rf_train)    
    X_rf_train_scaled = Scaler_Object_rf.transform(X_rf_train)
    X_rf_test_scaled = Scaler_Object_rf.transform(X_rf_test)
    
    Random_Forest_Object.fit(X_rf_train_scaled, y_rf_train)
    rf_reg_predictions = Random_Forest_Object.predict(X_rf_test_scaled)
    rf_reg_mse = mean_squared_error(y_rf_test, rf_reg_predictions)
    rf_reg_r2 = r2_score(y_rf_test, rf_reg_predictions)
    st.write('For random forest regression methods ', 'the accuracy score based on r2 ',np.round(rf_reg_r2),'.')
    st.write('For random forest regression methods ', 'the Mean Squared Error is  ',np.round(rf_reg_mse),'.')

    Index_rf=np.linspace(0,y_rf_test.size-1,y_rf_test.size).astype(int)
    RF_Dataframe=pd.DataFrame(index=np.arange(len(y_rf_test)), columns=np.arange(3))
    RF_Dataframe.columns=['Index','Actual','Predict']
    RF_Dataframe['Index'] = Index_rf
    RF_Dataframe['Actual'] = y_rf_test.reset_index(drop=True)
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
# Neural Network Regression
with tab5:
    Feature_Variable_DNN = st.multiselect('Select feature(s) for Neural Network Regression:',
                                        ['FFMC','DMC','DC','ISI','temp','RH','wind','rain'], default = 'temp')
    Target_Variable_DNN = df_forest_scaler['Logarea']
    Num_Hidden_Layer_DNN = st.selectbox('Number of Hidden Layers for NN Regression: ',(1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20), index=0)

    cols = st.columns(Num_Hidden_Layer_DNN)
    Num_Neuron_DNN=np.zeros(Num_Hidden_Layer_DNN)
    for j in range (Num_Hidden_Layer_DNN):
                with cols[j]:
                    Num_Neuron_DNN[j] = st.slider('Number of Neurons in '+str(j+1)+' Hidden Layer for regression:', min_value=1, max_value=45, value=10, step=1)
    Num_Neuron_DNN=Num_Neuron_DNN.astype(int)
    col1 , col2 , col3 , col4, col5= st.columns(5,gap='small')
    st.write(' ')
    Activation_DNN = col1.selectbox('Select activation function:',['identity', 'relu', 'logistic', 'tanh'],index = 0)
    Solver_DNN = col2.selectbox('Select solver type:',['adam', 'sgd', 'lbfgs'],index = 0)
    Alpha_DNN = col3.number_input('Input a non-negative value for alpha: ',value=0.01,format='%f')
    Learning_Rate_DNN = col4.selectbox('Select learning rate type:',['constant', 'invscaling', 'adaptive'],index = 0)
    Learning_Rate_Init_DNN = col5.number_input('Input a value for initial learning rate: ',value=0.001,format='%f')

    st.write(' ')
    col1 , col2 , col3 , col4, col5= st.columns(5,gap='small')
    st.write(' ')
    Validation_Fraction_DNN = col2.number_input('Input a value for validation fraction:',value=0.2,format='%f')
    Max_Iteration_DNN = col3.slider('Input a value for number of iteration:', 1, 20000, 200)
    Random_State_DNN = col5.slider('Input a value for DNN random state', 1, 200, 40)
    Tolerence_DNN = col1.number_input('Input a value for tolerence: ',value=0.0001,format='%f')
    Batch_Size_DNN = col4.slider('Input a value for batch size:', 1, len(y_test), 40)

    st.write(' ')
    col1, col2, col3= st.columns(3,gap='small')
    st.write(' ')
    Y_DNN = Target_Variable_DNN
    X_DNN = df_forest_scaler[Feature_Variable_DNN]
    Train_Size_DNN = col1.number_input('Input a value for train-size ratio:',value=0.8,format='%f')
    X_Train_DNN, X_Test_DNN, Y_Train_DNN, Y_Test_DNN = train_test_split(X_DNN, Y_DNN, train_size=Train_Size_DNN,random_state=42)
    
    Scaler_DNN = col2.checkbox('Applying Scaler object for neural network regression')
    if Scaler_DNN:
        Scaler_Type_DNN = col3.selectbox('Select NN scaler object:',['Min-Max Scaler', 'Standard Scaler'],index = 0)
        if Scaler_Type_DNN == 'Min-Max Scaler':
            Scaler_Object_DNN = MinMaxScaler()
        elif Scaler_Type_DNN == 'Standard Scaler':
            Scaler_Object_DNN = StandardScaler()
        Scaler_Object_DNN.fit(X_Train_DNN)
        X_Train_Scaled_DNN =Scaler_Object_DNN.transform(X_Train_DNN)
        X_Test_Scaled_DNN =Scaler_Object_DNN.transform(X_Test_DNN)
        
    MLP_Object=MLPRegressor(hidden_layer_sizes=Num_Neuron_DNN, activation=Activation_DNN, solver=Solver_DNN,
             alpha=Alpha_DNN, batch_size=Batch_Size_DNN, learning_rate=Learning_Rate_DNN,
             learning_rate_init=Learning_Rate_Init_DNN, max_iter=Max_Iteration_DNN, shuffle=True)
    
    if Scaler_DNN:
        MLP_Object.fit(X_Train_Scaled_DNN, Y_Train_DNN)
        score_DNN =MLP_Object.score(X_Test_Scaled_DNN, Y_Test_DNN)
        Y_Predic_DNN = MLP_Object.predict(X_Test_Scaled_DNN)
    else:
        MLP_Object.fit(X_Train_DNN, Y_Train_DNN)
        score_DNN =MLP_Object.score(X_Test_DNN, Y_Test_DNN)
        Y_Predic_DNN = MLP_Object.predict(X_Test_DNN)
        
    st.write(' ')
    st.markdown('<p class="font_text">Accuracy of the investigated (deep) neural network architecture:</p>', unsafe_allow_html=True)
    st.write(' ')
    Index_DNN=np.linspace(0,Y_Test_DNN.size-1,Y_Test_DNN.size).astype(int)
    DNN_Dataframe=pd.DataFrame(index=np.arange(len(Y_Test_DNN)), columns=np.arange(3))
    DNN_Dataframe.columns=['Index','Actual','Predict']
    DNN_Dataframe['Index'] = Index_DNN
    DNN_Dataframe['Actual'] = Y_Test_DNN.reset_index(drop=True)
    DNN_Dataframe['Predict'] = Y_Predic_DNN
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,1],marker_symbol='square',
                            mode='markers',
                            name='Actual '))
    fig2.add_trace(go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,2],marker_symbol='circle',
                            mode='markers',
                            name='Prediction '))
    st.plotly_chart(fig2)    
    st.markdown('<p class="font_text">Learning curve based on the above hyper-parameters:</p>', unsafe_allow_html=True)
####################################################################################################################################################################
# Support Vector Machines
with tab6:
    st.markdown('<p class="font_header"> Support Vector Machines:</p>', unsafe_allow_html=True)
    
    Feature_Variable_SVM = st.multiselect('Select feature(s) for SVR:',
                                        ['FFMC','DMC','DC','ISI','temp','RH','wind','rain'], default = 'temp')
    
    X_svm = df_forest_scaler[Feature_Variable_SVM]
    y_svm = df_forest_scaler['Logarea']
    X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(X_svm, y_svm, test_size=0.2)

    col1 , col2, col3,col4= st.columns(4,gap='small')
    Kernel_SVM = col1.selectbox('Choose the kernel type',['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],index = 2)
    CValue_SVM = col2.number_input('Input a value for C:',value=1,format='%f')
    epsilon_SVM = col3.number_input('Input a value for C:',value=0.1,format='%f')
    SVM_Object = SVR(kernel= Kernel_SVM, C = CValue_SVM, epsilon = epsilon_SVM)

    Scaler_Type_SVM = col4.selectbox('Select RF scaler object:',['Min-Max Scaler', 'Standard Scaler'],index = 1)
    if Scaler_Type_SVM == 'Min-Max Scaler':
        Scaler_Object_SVM = MinMaxScaler()
    elif Scaler_Type_SVM == 'Standard Scaler':
        Scaler_Object_SVM = StandardScaler()
    Scaler_Object_SVM.fit(X_svm_train)    
    X_svm_train_scaled = Scaler_Object_SVM.transform(X_svm_train)
    X_svm_test_scaled = Scaler_Object_SVM.transform(X_svm_test)
    
    SVM_Object.fit(X_svm_train_scaled, y_svm_train)
    SVM_reg_predictions = SVM_Object.predict(X_svm_test_scaled)
    svm_reg_mse = mean_squared_error(y_svm_test, SVM_reg_predictions)
    svm_reg_r2 = r2_score(y_svm_test, SVM_reg_predictions)
    st.write('For SVM regression methods ', 'the accuracy score based on r2 ',np.round(svm_reg_r2),'.')
    st.write('For SVM regression methods ', 'the Mean Squared Error is  ',np.round(svm_reg_mse),'.')

    Index_svm=np.linspace(0,y_svm_test.size-1,y_svm_test.size).astype(int)
    SVM_Dataframe=pd.DataFrame(index=np.arange(len(y_svm_test)), columns=np.arange(3))
    SVM_Dataframe.columns=['Index','Actual','Predict']
    SVM_Dataframe['Index'] = Index_svm
    SVM_Dataframe['Actual'] = y_svm_test.reset_index(drop=True)
    SVM_Dataframe['Predict'] = SVM_reg_predictions

    fig = go.Figure()
    fig3.add_trace(go.Scatter(x=SVM_Dataframe['Index'], y=SVM_Dataframe['Actual'],marker_symbol='square',
                        mode='markers',
                        name='Actual'))
    fig3.add_trace(go.Scatter(x=SVM_Dataframe['Index'], y=SVM_Dataframe['Predict'],marker_symbol='circle',
                        mode='markers',
                        name='Prediction'))

    st.plotly_chart(fig3)
    
#########################################################################################################################################
#Reference
st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Cortez, P., & Morais, A. D. J. R. (2007). A data mining approach to predict forest fires using meteorological data. </p>', unsafe_allow_html=True)
