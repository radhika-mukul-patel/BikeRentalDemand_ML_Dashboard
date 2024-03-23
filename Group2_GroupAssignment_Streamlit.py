import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
from PIL import Image
import time

# create page navigation
def page_navigation():
    pages = {
        "Executive Summary": page_summary,
        "Exploratory Data Analysis": page_eda,
        "Prediction Result": page_model,
        "Insights & Conclusion": page_conclusion,
        "Tech Annex":page_annex
    }
    st.sidebar.title("Bike Sharing Analysis & Prediction")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page()

# the first page is summary
def page_summary():
    st.title(":green[Bike Sharing Analysis & Prediction] _GroupB_  :bicyclist: :woman-biking:")
    st.image('https://bit.ly/3lhx9Lj')
    st.header("Executive Summary")
    st.subheader("Goal")
    st.markdown("<p style='font-size:18px'>Explore the bike rental behaviour in Washington DC to and predict the total numbers of bicycles that are rented/used every hour. These predictions and analysis will be used to optimise the bike service in the city.</p>", unsafe_allow_html=True)
    st.subheader("Techniques Implemented")
    st.markdown("- Exploratory Data Analysis")
    st.markdown("- Data Preprocessing (Label Encoding, Feature Importance)")
    st.markdown("- Pycaret")
    st.markdown("- Catboost Regressor")
    st.markdown("- Random Forest")
    st.markdown("- Model Ensemble")
    
#the second page is our plots
def page_eda():
    st.title(":green[Bike Sharing Analysis & Prediction] _GroupB_  :bicyclist: :woman-biking:")
    st.header("Exploratory Data Analysis")
    st.subheader("Information of the dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Columns", "17")
    col2.metric("Rows", "17379")
    col3.metric("Null Values", "0")
    st.subheader("First 5 rows of the dataset:")
    st.caption("Note: please refer to the Tech Annex for an explanation of the variables.")
    df = pd.read_csv('bike-sharing_hourly.csv')
    st.dataframe(df.head(), use_container_width=True)
    st.subheader("Analysis of the Bike Sharing Service")
    st.caption("Note: please refer to the Insights & Conclusion for further observations and perceptions.")
    st.markdown("Overview:")
    tab1 = st.tabs(["📕 Distribution by User"])
    # 1. Distribution of Rental Counts by type of user
    df = pd.read_csv('bike-sharing_hourly.csv')
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(y=["Registered", "Casual"], x=[df["registered"].sum(), df["casual"].sum()], orientation='h'))
    fig1.update_traces(opacity=0.75)
    fig1.update_layout(title="Distribution of rental counts by type of user", yaxis_title="User type", xaxis_title="Count")
    tab1[0].plotly_chart(fig1, use_container_width=True)
    
    st.markdown("By Time & Season:")
    df = pd.read_csv('bike-sharing_hourly.csv')
    tab2, tab3, tab4, tab5, tab6, tab10 = st.tabs(["📗 by Season", "📘 by Month", "📙 by Day", "📓 by Weekday/Weekend", "🎉 by Holiday", "⏱ by Hour"])
    # 2. Rental Counts by Season
    grouped_season = df.groupby("season").sum()["cnt"]
    fig2 = px.bar(grouped_season, title="Rental Counts by Season")
    fig2.update_layout(xaxis_title="Season", yaxis_title="Rental counts",
                      xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4],
                                 ticktext=['Winter', 'Spring', 'Summer', 'Fall']))
    # 3. Rental Counts by Month of the year
    grouped_month = df.groupby("mnth").sum()["cnt"]
    fig3 = px.bar(grouped_month, title="Rental Counts by Month")
    fig3.update_layout(xaxis_title="Month", yaxis_title="Rental counts",
                      xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                 ticktext=['January', 'February', 'March', 'April', 'May', 'June', 
                                           'July', 'August', 'September', 'October', 'November', 'December']))

    # 4. Rental Counts by Day of the week
    fig4 = px.bar(df, x="weekday", y=["casual", "registered"], title="Rental Counts by Day of the week", barmode="stack")
    fig4.update_layout(xaxis_title="Day of the week", yaxis_title="Rental counts",
                  xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6], 
                             ticktext=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']))

    # 5. Rental Counts by weekday or weekend
    grouped_weekday = df.groupby("workingday").sum()["cnt"]

    fig5 = px.bar(grouped_weekday, title="Rental Counts if the Day is a weekday or weekend")
    fig5.update_layout(xaxis_title="Weekday or Weekend?", yaxis_title="Rental counts",
                      xaxis=dict(tickmode='array', tickvals=[0, 1],
                                 ticktext=['Weekend', 'Weekday']))
    # Rental Counts by holiday
    grouped_holiday = df.groupby("holiday").sum()["cnt"]

    fig6 = px.bar(grouped_holiday, title="Rental Counts if the Day is a holiday")
    fig6.update_layout(xaxis_title="Holiday?", yaxis_title="Rental counts",
                      xaxis=dict(tickmode='array', tickvals=[0, 1],
                                 ticktext=['Non-holiday', 'Holiday']))
    # Rental Counts by Hours
    grouped_hours = df.groupby("hr").sum()["cnt"]
    fig10 = go.Figure()
    fig10.add_trace(go.Line(x=df["hr"], y=grouped_hours, mode='lines', name='Count of rental bikes'))
    fig10.update_layout(title="Rental Counts per hour",
                  xaxis_title="Hour of the day",
                  yaxis_title="Rental bike count",
                  xaxis=dict(tickmode='linear', tick0=0, dtick=1, tickvals=df["hr"], ticktext=[str(i) + ":00" for i in df["hr"]]))
    tab2.plotly_chart(fig2, use_container_width=True)
    tab3.plotly_chart(fig3, use_container_width=True)
    tab4.plotly_chart(fig4, use_container_width=True)
    tab5.plotly_chart(fig5, use_container_width=True)
    tab6.plotly_chart(fig6, use_container_width=True)
    tab10.plotly_chart(fig10, use_container_width=True)
    
    st.markdown("By Weather Condition:")
    df = pd.read_csv('bike-sharing_hourly.csv')
    

# 7. Rental Counts by weather situation
    df = pd.read_csv('bike-sharing_hourly.csv')
    tab7, tab8, tab9 = st.tabs(["🌀 by Weather", "🔆 by Temperature", "💦 by Humidity"])
    grouped_weather = df.groupby("weathersit").sum()["cnt"]
    fig7 = px.bar(grouped_weather, title="Rental Counts by Weather situation")
    fig7.update_layout(xaxis_title="Weather Situation", yaxis_title="Rental counts", 
                   xaxis=dict(tickmode='array', tickvals=[1,2,3,4], 
                              ticktext=['Clear, Few clouds, Partly cloudy', 
                                        'Mist + Cloudy, Mist + Broken clouds', 
                                        'Light Snow, Light Rain + Thunderstorm', 
                                        'Heavy Rain + Ice Pallets + Thunderstorm + Snow, Fog']))

# 8. Rental Counts by Temperature
    fig8 = px.scatter(df, x="temp", y="cnt", trendline="lowess", title="Rental Counts by Temperature", trendline_color_override="red")
    fig8.update_layout(xaxis_title="Temperature", yaxis_title="Rental counts")

# 9. Rental Counts by Humidity
    fig9 = px.scatter(df, x="hum", y="cnt", trendline="lowess", title="Rental Counts by Humidity", trendline_color_override="red")
    fig9.update_layout(xaxis_title="Humidity", yaxis_title="Rental counts")

    tab7.plotly_chart(fig7, use_container_width=True)
    tab8.plotly_chart(fig8, use_container_width=True)
    tab9.plotly_chart(fig9, use_container_width=True)
    # above is all about the second page

#--------------------------------------

# Auxiliary functions
def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3
    
def comf_hum (humidity_normalized):
    if humidity_normalized >= 0.25 and humidity_normalized <= 0.55:
        return 1
    else:
        return 0
    
def comf_temp (temp_normalized):
    if temp_normalized >= 0.40 and temp_normalized <= 0.65:
        return 1
    else:
        return 0
    
def hour_to_time_of_day(hour):
    time_of_day_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 0,
        22: 0,
        23: 0
    }
    return time_of_day_dict[hour]

#the third page is about the model
def page_model():
    model = pickle.load(open("bikeRentalsModel.pkl", "rb"))
    df = pd.read_csv("preprocessed-df.csv")
    
    weather_values = ['Clear, Few clouds, Partly cloudy, Partly cloudy',
                      "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
                      "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
                      "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"]
    weekday_values = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    boolean_values = ["No", "Yes"]
    year_values = ["2011", "2012"]
    month_values = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    st.title(":green[Bike Sharing Analysis & Prediction] _GroupB_  :bicyclist: :woman-biking:")
    st.subheader("Enter the following information to predict the number of bike rentals:")
    st.markdown('Time-wise:') 
    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox('Year', options=[0, 1], format_func=lambda x: year_values[x])
        month = st.selectbox('Month', options=range(1, 13), format_func=lambda x: month_values[x-1])
        holiday = st.selectbox('Holiday', options=[0, 1], format_func=lambda x: boolean_values[x])
        st.markdown('Weather-wise:') 
    with col2:
        day = st.selectbox('Day', options=range(1, 31))
        hour = st.selectbox('Hour', options=range(0, 24))
        weekday = st.selectbox('Weekday', options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: weekday_values[x])
        st.header('  ') 
           
        
    with col1:
        weather = st.selectbox('Weather', options=[1, 2, 3, 4], format_func=lambda x: weather_values[x-1])
        conv_factor2 = 100.0
        humidity = st.slider('Humidity (%)', min_value=0, max_value=int(conv_factor2), step=1, value=int(0.5*conv_factor2), format='%d %%')
        humidity_normalized = humidity / conv_factor2

    with col2:
        conv_factor = 41.0
        temp = st.slider('Temperature (°C)', min_value=0, max_value=int(conv_factor), step=1, value=int(0.5*conv_factor), format='%d °C')
        temp_normalized = temp / conv_factor
        conv_factor1 = 67.0
        windspeed = st.slider('Wind Speed (km/h)', min_value=0, max_value=int(conv_factor1), step=1, value=int(0.5*conv_factor1), format='%d km/h')
        windspeed_normalized = windspeed / conv_factor1

        
        
                
    input_data = {
        'season': get_season(month),
        'yr': year,
        'mnth': month,
        'hr': hour,
        'holiday': holiday,
        'weekday': weekday,
        'workingday': 1 - holiday,
        'weathersit': weather,
        'atemp': temp_normalized,
        'hum': humidity_normalized,
        'windspeed': windspeed_normalized,
        'day' : day,
        'time_of_day' : hour_to_time_of_day(hour),
        'comfortable_temp' : comf_temp(temp_normalized),
        'comfortable_humidity' : comf_hum(humidity_normalized)
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)
    
    rounded_prediction = round(prediction[0])
    
    st.subheader('Predicted Number of Bike Rentals:')
    st.markdown(f"<p style='font-size:30px; color:green'>{rounded_prediction}</p>", unsafe_allow_html=True)
    bike_emoji = '<span style="font-size:30px"> 🚲 </span>'
    st.write(bike_emoji * rounded_prediction, unsafe_allow_html=True)
    
#the forth page is about the insights
def page_conclusion():
    st.title(":green[Bike Sharing Analysis & Prediction] _GroupB_  :bicyclist: :woman-biking:")
    st.header("Insights & Conclusion")
    tab1, tab2 = st.tabs(["💡 Insights", "📌 Conclusion"])
    
    with tab1:
        st.subheader('User Type:')
        st.markdown("<p style='font-size:18px'> The first important factor to explore is the proportion of casual vs. registered bike riders in DC. There are 2.7M registered bikers and 620K casual riders. These proportions are consistent with our expectations, as the registered bikers eventually end up paying less per ride with their membership. In addition, registered drivers are more likely to use their bikes for commuting and as a transportation method for everyday activities such as grocery shopping.  </p>", unsafe_allow_html=True)
        st.subheader('Season & Weather:')
        st.markdown("<p style='font-size:18px'> As expected, people in DC rent more bikes during the summer season as the weather is more appropriate for outdoor activities. In fact, twice the amount of people ride bikes in Summer, with 1.06M riders, compared to Winter with 471K riders.   </p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px'> During the summer months, from June to September, the number of bike riders is steady with around 340K to 350K riders per month. The month that has the lowest riders is January with 135K. This is to be expected as it is one of the coldest months in DC, and the snow and wind can be a hazard for bike drivers, especially those with less experience.   </p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px'> Related to the season, most people in DC rent a bike when the weather is clear up to partly cloudy, as this is the safest and most enjoyable weather. There were 2.34M bike rentals registered for this kind of weather, compared to 955K for other kinds of weather including cloudy, snowy or rainy.   </p>", unsafe_allow_html=True)
        st.subheader('Temperature & Humidity:')
        st.markdown("<p style='font-size:18px'> Regarding the temperature and humidity, bike rentals are highest when these factors are optimal for outdoor activities. With temperature, we can see a positive correlation between bike rentals and temperature in DC. Worthy to note that at the highest temperatures, the bike rentals decrease because the weather is too hot, as it becomes uncomfortable and potentially dangerous.   </p>", unsafe_allow_html=True)
        st.subheader('Day of the Week & Hour:')
        st.markdown("<p style='font-size:18px'> The number of riders throughout the week is also fairly similar. However, we see that the proportion of casual bike riders is higher on Saturday and Sunday. This is to be expected as, probably, more registered riders will rent a bike during the week, for their commute, while casual riders rent a bike during the weekend as a recreational activity.  </p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px'> Surprisingly, Sunday is the day with the lowest number of bike riders, with 444K riders. Friday is the day with the most riders with 488K. This could be because Friday is a work day, so registered drivers will still commute to work, for some casual riders also rent bikes for leisure.   </p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px'> When checking the amount of bikes rented per hour, our past predictions are confirmed as the most popular times for bike rentals are around 8 am and  5 am, which are the usual times for commuting to work in DC. In addition, there is a small spike at 12 pm, as bikers likely rent a bike to get lunch outside of their work or home. The times with the least bike rentals are 11 pm to 5 am, which is expected as DC citizens are asleep.   </p>", unsafe_allow_html=True)
    with tab2:
        st.subheader('Focus on Optimal Bike Repair and Maintenance')
        st.markdown("<p style='font-size:18px'> To provide an optimal service to the community of DC, we recommend for the bikes to be fixed or maintained during the winter months, when people ride less, so they are ready for the summer months when there is a high demand. Any emergency maintenance or repair should be done between 11 pm and 5 am for the bikes or the station themselves. Finally, the city could include a site for quick and free repairs next to bike stations in areas where DC citizens usually work, such as the financial district area.   </p>", unsafe_allow_html=True)
        st.subheader('Educate Riders about Winter Safety')
        st.markdown("<p style='font-size:18px'> Creating a campaign with advertisements around the city (commercials or posters around the bike stations) providing tips for bike riders to safely bike in the winter months and when the weather conditions could be dangerous (ex: snow, rain, fog). The campaign’s effort should specially be targeted around November to January, when the winter is starting and intensifying.  </p>", unsafe_allow_html=True)
        st.subheader('Winter Discounts to Incentivize Commuters')
        st.markdown("<p style='font-size:18px'> Coupled with the campaign to provide tips for safe riding techniques, the city of DC could implement a special discount for registered drivers during the winter months so they are incentivized to still utilize bikes for commuting. This will ensure a  constant flow of revenue coming in each month and we can have better predictions.   </p>", unsafe_allow_html=True)
        st.subheader('Improve Infrastructure to Create a Bike-Friendly City')
        st.markdown("<p style='font-size:18px'> In order for casual riders to become registered drivers, they need to feel like it’s safe and worth it to rent a bike for their everyday activities, compared to using other methods of transportation. These tactics include improvement in the comfort and safety of bicycle lanes around the city. Moreover, the whole road infrastructure (e.g. junctions, crossings, traffic signs, etc) needs to be altered to give cyclists priority.  </p>", unsafe_allow_html=True)
        st.subheader('Focus on Low-income Neighborhoods')
        st.markdown("<p style='font-size:18px'> For improvements around the city, DC officials should also focus on making low-income neighborhoods more bike accessible, as citizens living in these neighborhoods are less likely to be able to afford cars or the metro. This will provide them the flexibility for an easier commute and will help save money for other investments. Changes can include adding bike stations and sites for quick and free repairs, as mentioned above.  </p>", unsafe_allow_html=True)
        st.subheader('Incentivize Women to Bike to Work')
        st.markdown("<p style='font-size:18px'> Create a program that incorporates workshops, rides, and mentoring programs to draw women to cycling. This is because women usually make up a smaller percentage of the city bike commuters compared to men. Therefore, women could also receive special discounts when they rent bikes during commute hours, as specified in our findings. In addition, to incentivize women to ride bikes more often, the city of DC has to constantly monitor that the majority of bikes have a proper basket, as women tend to carry purses and other items.  </p>", unsafe_allow_html=True)
        st.subheader('Gather Data to See What are The Most Popular Routes')
        st.markdown("<p style='font-size:18px'> For future recommendations, we suggest gathering data about the most common routes taken by registered cyclists and those used by people who commute (those renting bikes around 8 am and/or 5 pm). Then, the city of DC should use this data in order to prioritize which parts of the city needs a new or improved bike lane. In addition, any campaign’s propaganda, as the one mentioned above for safety, should be concentrated on these areas.  </p>", unsafe_allow_html=True)
	
#the fifth page is about the result
def page_annex():
    st.title(":green[Bike Sharing Analysis & Prediction] _GroupB_  :bicyclist: :woman-biking:")
    st.header("Tech Annex")
    tab1, tab2 = st.tabs(["📈 Model Accuracy", "📋 Variable Explanation"])
    with tab1:
        st.subheader("Model Accuracy:")
        col1, col2 = st.columns(2)
        col1.metric("R2 Score", "0.86")
        col2.metric("Almost Correct Predictions", "90.13%")
        model = pickle.load(open("bikeRentalsModel.pkl", "rb"))
        df = pd.read_csv("preprocessed-df.csv")
        cutoff_date = pd.to_datetime('2012-09-01')
        df["dteday"] = pd.to_datetime(df["dteday"])
        train = df.loc[df.dteday < cutoff_date]
        test = df.loc[df.dteday >= cutoff_date]

        X_train, y_train, X_test, y_test = train.drop(["cnt", "dteday"], axis=1), train["cnt"], test.drop(["cnt", "dteday"], axis=1), test["cnt"]
        predictions = model.predict(X_test)

        # Creating the line chart with Plotly Express
        fig10 = px.line(train, x='dteday', y='cnt', title='Actual vs Predicted Bike Rentals')
        actual_trace = go.Scatter(x=test["dteday"], y=predictions, name='Prediction', line=dict(color='#FF6433'))
        fig10.add_trace(actual_trace)
        st.plotly_chart(fig10)

        # Creating the line chart with Plotly Express
        fig11 = px.line(x=test['dteday'], y=predictions, title='Actual vs Predicted Bike Rentals (Zoomed-In)')
        # Creating a new trace for the actual values
        actual_trace = go.Scatter(x=df['dteday'], y=df['cnt'], name='Actual', line=dict(color='#1EB33D'))
        # Adding the actual and mean error traces to the figure
        fig11.add_trace(actual_trace)
        # Setting the range for the x-axis to show a zoomed in region
        fig11.update_xaxes(range=['2012-12-01 00:00:00', '2012-12-31 00:00:00'])
        st.plotly_chart(fig11)
    with tab2: 
        st.subheader("Variable Explanation:")
        st.write("""- `instant`: record index
- `dteday` : modified to date + `hr`
+ `season` : season
	- 1: Spring
	- 2: Summer
	- 3: Fall
	- 4: Winter
- `yr` : year (0: 2011, 1:2012)
- `mnth` : month ( 1 to 12)
- `day` : extracted the day from `dteday`
- `hr` : hour (0 to 23)
- `time_of_day` : aggregated `hr` to "night", "morning", "afternoon", "evening"
- `holiday` : whether day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
- `weekday` : day of the week
- `workingday` : if day is neither weekend nor holiday is 1, otherwise is 0.
+ `weathersit` : 
	- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp` : Normalized temperature in Celsius. The values are divided to 41 (max)
- `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- `hum`: Normalized humidity. The values are divided to 100 (max)
- `windspeed`: Normalized wind speed. The values are divided to 67 (max)
- `comfortable_temp` : if temperature is comfortable(`atemp`>0.4), it's 1, otherwise is 0.
- `comfortable_humidity` : if humidity is comfortable(0.55>`hum`>0.25), it's 1, otherwise is 0.
- `casual`: count of casual users
- `registered`: count of registered users
- `cnt`: count of total rental bikes including both casual and registered""")

    
# Run the app
if __name__ == "__main__":
    page_navigation()
