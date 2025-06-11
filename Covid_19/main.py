import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyAV4-9fv3ltLsPKySpHeRTojdyzr_BXG_o")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

st.set_page_config(
    page_title="Covid-19 Dashboard",
    layout="wide"
)



df_recover = pd.read_csv('https://raw.githubusercontent.com/raj-coding1/Covid19/refs/heads/main/Datasets/covid_19_recovered_v1_lyst1747728719904.csv')
df_confirm = pd.read_csv('https://raw.githubusercontent.com/raj-coding1/Covid19/refs/heads/main/Datasets/covid_19_confirmed_v1_lyst1747728690432.csv')
df_death = pd.read_csv('https://raw.githubusercontent.com/raj-coding1/Covid19/refs/heads/main/Datasets/covid_19_deaths_v1_lyst1747728711771.csv')
#initial preprocess
df_recover.columns = df_recover.iloc[0]
df_recover = df_recover[1:].reset_index()
df_recover.drop('index',axis = 1,inplace = True)

df_death.columns = df_death.iloc[0]
df_death = df_death[1:].reset_index()
df_death.drop('index',axis = 1,inplace = True)


# df_recover
# df_confirm
# df_death

#melt
df_confirm_melt = df_confirm.melt(id_vars = ['Province/State','Country/Region','Lat','Long'],var_name = 'Date',value_name = 'Confirm')
df_confirm_melt['Date'] = pd.to_datetime(df_confirm_melt['Date'])
df_confirm_melt.set_index('Date',inplace = True)
# df_confirm_melt = df_confirm_melt[(df_confirm_melt['Lat']!=0) & (df_confirm_melt['Long']!=0)]
df_confirm_melt['Country/Region']= df_confirm_melt['Country/Region'].replace('US','USA')


min_date = df_confirm_melt.index.min()
max_date = df_confirm_melt.index.max()
def gr(from1,to,df,status,country):
    if country!='World':
        tem_df = df.loc[to]
        temx_df = tem_df[tem_df['Country/Region']==country]

    else:
        tem_df = df.loc['2021-5-29']
    # # tem_df.groupby('Country/Region')['Confirm']
    grouped = tem_df.groupby('Country/Region', as_index=False).agg({
        status: 'sum',
        'Lat': 'first',
        'Long': 'first'

    }).sort_values('Confirm',ascending = False)
    return grouped
# df_confirm_melt

df_recover_melt = df_recover.melt(id_vars = ['Province/State','Country/Region','Lat','Long'],var_name = 'Date',value_name = 'Recover')
df_recover_melt['Date'] = pd.to_datetime(df_recover_melt['Date'])
df_recover_melt.set_index('Date',inplace = True)
df_recover_melt['Country/Region']= df_recover_melt['Country/Region'].replace('US','USA')
# df_recover_melt = df_recover_melt[(df_recover_melt['Lat']!=0) & (df_recover_melt['Long']!=0)]

df_death_melt = df_death.melt(id_vars = ['Province/State','Country/Region','Lat','Long'],var_name = 'Date',value_name = 'Death')
df_death_melt['Date'] = pd.to_datetime(df_death_melt['Date'])
df_death_melt.set_index('Date',inplace = True)
df_death_melt['Country/Region']= df_death_melt['Country/Region'].replace('US','USA')
# df_death_melt = df_death_melt[(df_death_melt['Lat']!=0) & (df_death_melt['Long']!=0)]
df_confirm_melt['Confirm'] = df_confirm_melt['Confirm'].astype('float64')
df_death_melt['Death'] = df_death_melt['Death'].astype('float64')
df_recover_melt['Recover'] = df_recover_melt['Recover'].astype('float64')
# df_confirm_melt = df_confirm_melt[(df_confirm_melt['Lat']>0) & (df_confirm_melt['Long']>0)]
df = df_confirm_melt.groupby([df_confirm_melt.index,'Country/Region']).agg(
    {
        'Confirm' : 'sum',
        'Lat' : 'first',
        'Long' : 'first',
    }
).reset_index()
df_re = df_recover_melt.groupby([df_recover_melt.index,'Country/Region']).agg(
    {
        'Recover' : 'sum',
        'Lat' : 'first',
        'Long' : 'first',
    }
).reset_index()
df_de = df_death_melt.groupby([df_death_melt.index,'Country/Region']).agg(
    {
        'Death' : 'sum',
        'Lat' : 'first',
        'Long' : 'first',
    }
).reset_index()

df[['Lat', 'Long']] = df_re[['Lat', 'Long']].values
df_de[['Lat', 'Long']] = df_re[['Lat', 'Long']].values

x = df.merge(df_re,on = ['Country/Region','Lat','Long','Date'],how = 'outer')
merge_df = x.merge(df_de,on = ['Country/Region','Lat','Long','Date'],how = 'outer')
merge_df['Date'] =pd.to_datetime(merge_df['Date']) 
merge_df.set_index('Date',inplace = True)
da_tem = merge_df[merge_df['Country/Region'] == 'USA']
tem_da = da_tem.loc['2020-12-13']
rr = round(tem_da['Recover']/tem_da['Confirm']*100,2)
# st.write(rr)
temporary_da = da_tem.loc['2020-12-14':'2021-5-29']
temporary_da['Recover'] = round((temporary_da['Confirm']*rr)/100,0)
da_tem.loc['2020-12-14':'2021-5-29'] = temporary_da
merge_df[merge_df['Country/Region'] == 'USA'] = da_tem
# merge_df
# da[da['Country/Region'] == 'USA']
def compute_features(df):
    df = df.sort_values('Date').copy()

    df['Single_Death'] = df['Death'].diff()
    df['Single_Confirm'] = df['Confirm'].diff()
    df['Single_Recover'] = df['Recover'].diff()

    df['7Day_Death'] = df['Single_Death'].rolling(window=7).mean()
    df['7Day_Confirm'] = df['Single_Confirm'].rolling(window=7).mean()
    df['7Day_Recover'] = df['Single_Recover'].rolling(window=7).mean()

    return df
merge_df = merge_df.reset_index()
merge_df['Date']
da = merge_df.groupby('Country/Region', group_keys=False).apply(compute_features).reset_index(drop=True)
da[['Lat','Long']] = da[['Lat','Long']].astype('float64')
# da
# da['Date'] = pd.to_datetime(da['Date'])
# import pandas as pd
import pycountry_convert as pc

da['Country/Region']= da['Country/Region'].replace('US','USA')
da = da[(da['Lat']!=0) & (da['Long']!=0)]
da.fillna(0,inplace = True)
# st.write(da.isnull())
df_confirm_melt['Country/Region']= df_confirm_melt['Country/Region'].str.replace('US','USA')
# Helper function to get continent from country
# Helper function to convert country to continent
def country_to_continent(country_name):
    try:
        # Convert country name to ISO alpha-2 code (like 'IN', 'US')
        country_code = pc.country_name_to_country_alpha2(country_name)
        # Convert ISO code to continent code (like 'AS', 'EU')
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        # Map continent code to name
        continent_name = {
            'AF': 'Africa',
            'AS': 'Asia',
            'EU': 'Europe',
            'NA': 'North America',
            'SA': 'South America',
            'OC': 'Oceania',
            'AN': 'Antarctica'
        }[continent_code]
        return continent_name
    except:
        return 'Unknown'


# da[da['Single_Confirm']<0]

# da
da['Continent'] = da['Country/Region'].apply(country_to_continent)
df_confirm_melt['Province/State'].fillna('All Province',inplace = True)
df_confirm_melt['Continent'] = df_confirm_melt['Country/Region'].apply(country_to_continent)
# print(df)
# x = df_confirm_melt.loc['2021-5-29']
# x
da['Month'] = da['Date'].dt.to_period("M")
monthly_df = da.groupby(['Month','Country/Region']).tail(1)
# monthly_df
da.set_index('Date',inplace = True)
x = da.loc['2021-5-29']
top20_df = (
    x.groupby('Continent', group_keys=False)
      .apply(lambda x: x.nlargest(10, 'Confirm'))
)
da['Recovery_rates'] = round((da['Recover']/da['Confirm'])*100,2)
da['Death_rates'] = round((da['Death']/da['Confirm'])*100,2)
# da['Recovery_rates'] = round((df['Recover']/df['Confirm'])*100,2)
da.fillna(0,inplace = True)
# da
## Create Variable for AI
da['Active'] = da['Confirm'] - (da['Death'] + da['Recover'])
# tem_df = da.loc['2021-5-29']



da_tem = da[da['Country/Region'] == 'USA']
tem_da = da_tem.loc['2020-12-13']
rr = round(tem_da['Recover']/tem_da['Confirm']*100,2)
# st.write(rr)
temporary_da = da_tem.loc['2020-12-14':'2021-5-29']
temporary_da['Recover'] = round((temporary_da['Confirm']*rr)/100,0)
da_tem.loc['2020-12-14':'2021-5-29'] = temporary_da
da[da['Country/Region'] == 'USA'] = da_tem
da[da['Country/Region'] == 'USA']







monthly_confirm = da.loc[pd.date_range('2020-1-30','2021-5-29',freq = pd.Timedelta(days = 30))]
gr_coun_date = monthly_confirm.groupby([monthly_confirm.index,'Country/Region'])[['Confirm','Recover','Death']].sum()
# gr_coun_date
gr_coun_date = gr_coun_date.reset_index().sort_values('Confirm',ascending = False)
# top20_df

# grouped['Country/Region'] = grouped['Country/Region'].str.replace('US','USA')
# da = da.set_index('Date')
# df_confirm_melt
# st.write(df_recover)
# st.write(df_confirm)
# st.write(df_death)

all_countries = da['Country/Region'].unique()
# st.write(all_countries)
all_countries= np.insert(all_countries,0,'World')

st.sidebar.title('Customizer')
country = st.sidebar.selectbox('Choose the Country',options = all_countries)
st.title(country+' (Covid19) Dashboard')
da['label'] = da['Country/Region'].apply(lambda x: x if x == country else '')
# grouped
monthly_df.set_index('Date',inplace=True)
center_lat = 0
center_lon = 0
if country!= 'World':
    selected_row = da[da['Country/Region'] == country].iloc[0]
    center_lat = selected_row['Lat']
    center_lon = selected_row['Long']
st.sidebar.write('Select a date')
from1 = st.sidebar.date_input('From',value = min_date)
to = st.sidebar.date_input('To',value = max_date)
from2 = pd.to_datetime(from1)
to1 = pd.to_datetime(to)

from_str = from1.strftime('%Y-%m-%d')
to_str = to.strftime('%Y-%m-%d')

Status = st.sidebar.selectbox('Choose Status', options = ['Confirm','Recover','Death','Overall Analysis'])
tem_df = da.loc[to1]
last_24_confirm = tem_df['Confirm'].sum()
total_confirm = tem_df['Confirm'].sum()
total_death = tem_df['Death'].sum()
total_recover = tem_df['Recover'].sum()
# total_active = tem_df['Active'].sum()
total_active = (total_confirm - (total_death+total_recover))
max_spike_death = da['Single_Death'].max()
max_spike_confirm = da['Single_Confirm'].max()
max_spike_recover = da['Single_Recover'].max()
death_rate = round((total_death/total_confirm)*100,2)
recovery_rate = round((total_recover/total_confirm)*100,2)
# last_24_confirm = da['S']

st.markdown(
f"""
<div style="display: flex; justify-content: space-between; width: 100%; font-size: 18px;">
    <div><strong>Analysis for:</strong> {country}</div>
    <div><strong>From:</strong> {from_str} &nbsp;&nbsp;&nbsp; <strong>To:</strong> {to_str}</div>
</div>
""",
unsafe_allow_html=True
)
st.divider()
def RateMetric(from1,to,df_rate,country):
    if country != 'World':
        death_rate = round((df_rate.loc[to]['Death']/df_rate.loc[to]['Confirm'])*100,2)
        recover_rate = round((df_rate.loc[to]['Recover']/df_rate.loc[to]['Confirm'])*100,2)
        MetricDesign(death_rate,'','Death Rate')
        MetricDesign(recover_rate,'','Recover Rate')
        

    # df_rate = df.groupby('Country/Region')[['Confirm','Recover','Death']].sum()
    # df_rate['Death_rate']= round((df_rate['Death'] / df_rate['Confirm'])*100,2)
    # df_rate['Recover_rate']= round((df_rate['Recover'] / df_rate['Confirm'])*100,2)
    # (df_rate['Death_rate'].sum())/len(df_rate)
    # if country != 'World':
    #     df_rate = df_rate[df_rate['Country/Region']== country].loc[]
    else:
        highest_death_rate = df_rate['Death_rate'].max()
        highest_death_rate_Country = df_rate[df_rate['Death_rate'] == highest_death_rate].index
        lowest_death_rate = df_rate['Death_rate'].min()
        lowest_death_rate_Country = df_rate[df_rate['Death_rate'] == lowest_death_rate].index
        # lowest_death_rate_Country
        highest_recover_rate = df_rate['Recover_rate'].max()
        highest_recover_rate_Country = df_rate[df_rate['Recover_rate'] == highest_recover_rate].index

    
        MetricDesign(highest_death_rate,highest_death_rate_Country[0],'highest death rate ')
        MetricDesign(highest_recover_rate,highest_recover_rate_Country[0],'highest recover rate ')
        # MetricDesign(highest_confirm_cases,highest_confirm_cases_country,'highest confirm cases ')




    
def Donut(df_rate,country):
    # if country != 'World':

        
    total_confirm = df_rate['Confirm'].sum()
    total_recover = df_rate['Recover'].sum()
    total_death = df_rate['Death'].sum()
    param = ['confirm','recover','death']
    value = [total_confirm,total_recover,total_death]
    sum_data = pd.DataFrame({
        'Category' : param,
        'Values' : value,
    })
    fig = px.pie(sum_data, names='Category', values='Values', hole=0.4)  
    st.plotly_chart(fig)

def HighestCases(df_rate,country):
    # df_rate = df_rate.loc['2021/']
    if country != 'World':
        df_rate = df_rate[df_rate['Country/Region']== country]
    
    highest_death_cases = df_rate['Death'].max()
    highest_recover_cases = df_rate['Recover'].max()
    highest_confirm_cases = df_rate['Confirm'].max()
    highest_death_cases_country = df_rate[df_rate['Death'] == highest_death_cases].index[0]
    highest_recover_cases_country = df_rate[df_rate['Recover'] == highest_recover_cases].index[0]
    highest_confirm_cases_country = df_rate[df_rate['Confirm'] == highest_confirm_cases].index[0]
    MetricDesign(highest_recover_cases,highest_recover_cases_country,'highest recover cases ')
    MetricDesign(highest_death_cases,highest_death_cases_country,'highest death cases ')
    MetricDesign(highest_confirm_cases,highest_confirm_cases_country,'highest confirm cases ')
def MetricDesign(value_confirm,arrow,label):
    
    label = label
    try:
        value = f"{int(float(value_confirm)):,}"
    except (ValueError, TypeError):
        value = "N/A"

    # value = value_confirm
    # delta = delta_confirm
    arrow = arrow
    # "↑" 
    color = "green"
    # st.metric(label,value,border = True)
    st.markdown(
    f"""
    <div style="border: 1px solid #ddd; border-radius: 0.5rem; padding: 1rem; text-align: center">
        <div style="font-size: 1rem; color: gray;">{label}</div>
        <div style="font-size: 1.7rem; font-weight: bold;">{value}</div>
        <div style="font-size: 0.9rem; color: {color};">{arrow} </div>
    </div>
    """,
    unsafe_allow_html=True
)
def geoScatter(df,status,country,color_scale,center_lon,center_lat):
    if status == 'Overall Analysis':
        status = 'Confirm'
    fig = px.scatter_geo(df,
                                lat='Lat',
                                lon='Long',
                            
                                color=status,       # Bubble color = confirmed cases
                                color_continuous_scale=color_scale,
                                hover_name='Country/Region',
                                # scope = 'africa',
                                text = 'label',
                                projection = 'natural earth',
                                hover_data={'Country/Region': True, 'Lat': False,'Long': False},
                                # title='COVID-19 Hotspots by Country'
                                )

    if country != 'World':
        fig.update_traces(textposition='top center', textfont=dict(size=15, color='red',family = 'Arial'))
        fig.update_geos(
        showland=True, landcolor="lightgray",
        showcountries=True,
        showocean=True, oceancolor="lightblue",
        center=dict(lat=center_lat, lon=center_lon),  # Center the view
        lonaxis_range=[center_lon - 60, center_lon + 40],  # Horizontal zoom
        lataxis_range=[center_lat - 20, center_lat + 20],  # Vertical zoom
        )
    else:
         fig.update_traces(textposition='top center', textfont=dict(size=12, color='black'))

            # Improve map aesthetics
         fig.update_geos(
            showland=True,
            landcolor="lightgray",
            showcountries=True,
            showocean=True,
            
            oceancolor="lightblue",
            showframe=True,
            framecolor="black"
        )

    
    

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    # fig.update_layout(geo=dict(showland=True, landcolor='white'))
    
    st.header("COVID-19 Map For "+ country)
    st.plotly_chart(fig,use_container_width=True)
    

def TotalCases(from1,to,df,status,country):
    # mask = (df.index >= from1) & (df.index <= to)
    # df = df[mask]
    if country == 'World':
        total = df.loc[to][status].sum()
    else:
        df = df[df['Country/Region'] == country]
        total = df.loc[to][status].sum()
    MetricDesign(int(total),"Δ",status)

def maxSpike(from1,to,df,status,country):
    # mask = (df.index >= from1) & (df.index <= to)
    # df = df[mask]
    if country == 'World':
        val = df[status].max()
        # print()
        # df[df[status]== val]['Country/Region']
    else:
        df = df[df['Country/Region'] == country]
        val = df[status].max()
    status = 'Single_Day_Max_Spike'
    MetricDesign(int(val),"Δ",status)

def OneDay(from1,to,df,status,country):

    if country == 'World':
        val = df.loc[to][status].sum()
    else:
        df = df[df['Country/Region'] == country]
        val = df.loc[to][status]
    status = 'Last 24hr '+status
    MetricDesign(int(val),"Δ",status)
    
  
def SDay(from1,to,df,status,country):
    if country == 'World':
        val = df.loc[to][status].sum()
    else:
        df = df[df['Country/Region'] == country]
        val = df[status].sum()
    MetricDesign(int(val),"Δ",'7 day moving average ')

def Active_cases(from1,to,df,status,country):
    if country == 'World':
        df = df.loc[to]
        val = df['Confirm'].sum() - (df['Death'].sum() + df['Recover'].sum())
    else:
        df = df[df['Country/Region'] == country]
        df = df.loc[to]
        val = df['Confirm'].sum() - (df['Death'].sum() + df['Recover'].sum())

    MetricDesign(int(val),"Δ",status)

# tab1,tab2 = st.tabs(['Overview','Maps'])
    
def MetricStruct(from2,to1,df_confirm_melt,status,country):
    
    col1,col2,col3,col4 = st.columns([1,1,1,1])
    with col1:
        if status == 'Overall Analysis':
            r = 'Confirm'
            TotalCases(from2,to1,df_confirm_melt,r,country)
        else:
            TotalCases(from2,to1,df_confirm_melt,status,country)
        # MetricDesign(value_confirm,delta_confirm)

    with col2:
        
        if status == 'Overall Analysis':
            a = 'Death'
            TotalCases(from2,to1,df_confirm_melt,a,country)


        else:
            x = 'Single_'+status
            maxSpike(from2,to1,df_confirm_melt,x,country)
        # MetricDesign(value_recover,delta_recover)
        
    with col3:
        
        if status == 'Overall Analysis':
            j = 'Recover'
            TotalCases(from2,to1,df_confirm_melt,j,country)
        else:
            x = 'Single_'+status
            OneDay(from2,to1,df_confirm_melt,x,country)
        # MetricDesign(value_death,delta_death)
    with col4:
        
        if status == 'Overall Analysis':
            k = 'Active'
            Active_cases(from2,to1,df_confirm_melt,k,country)
        else:
            y = '7Day_'+status
            SDay(from2,to1,df_confirm_melt,y,country)
    # with col5:
    #     x = 'Single_'+status
        
    #     OneDay(from2,to1,df_confirm_melt,x,country)
def SunPlot(df,status):
    col1,col2 = st.columns([3,1])
    with col1:
        top20_df = (
        df.groupby('Continent', group_keys=False)
        .apply(lambda x: x.nlargest(10, Status))
        )
        plt.figure(figsize=(15,15))
        
        fig = px.sunburst(df,path= ['Continent','Country/Region'],values = status,color='Continent',  # use this column to assign color
        color_discrete_map={
            'Asia': 'lightblue',
            'Europe': 'lightgreen'
        }) 
        fig.update_layout(width=500, height=600)
        fig.update_traces(textfont_size=18)

        # st.title("🌍 COVID-19 Global Map")
        st.plotly_chart(fig)

def Rate(from1,to,df,country,status):
    if country == 'World':
        df = df.loc[to]
        val = round((df[status].sum() / df['Confirm'].sum())*100,2)
    else:
        df = df[df['Country/Region'] == country]
        df = df.loc[to]
        val = (df[status].sum() / df['Confirm'].sum())*100
    MetricDesign((val),'',status+'_Rate of'+country)

tab1,tab2 = st.tabs(['Data','Data AI'])
with tab1:
    if Status == 'Overall Analysis':
        # 'hello'
        

        if country == 'World':
            df = da
            # grouped = gr(from2,to1,df,Status,country)
            mask = (da.index >= from2) & (da.index <= to1)
            
            df1 = da[mask]
            df_rate = df1[df1['Confirm']>3000000].loc[to1]
            df_rate['Recover_rate'] = round((df_rate['Recover']/df_rate['Confirm'])*100,2)
                # st.write(data.sort_values('Recovery_Rate',ascending = False))
            df_rate['Death_rate'] = round((df_rate['Death']/df_rate['Confirm'])*100,2)
            # df1
            MetricStruct(from2,to1,df1,Status,country)
            df2 = df1.loc[to1]
            top_10 = df2.sort_values('Confirm',ascending = False).head(10)['Country/Region'].unique()
            line_df = df[df['Country/Region'].isin(top_10)]
            grouped = df1.loc[to1]
            bar_df = line_df.loc[pd.date_range(from2,to1,freq = '31D')].sort_values('Confirm',ascending = False)
            
            
            # with tab1:
            color_scale = [
                [0.0, "green"],
                [0.25, "yellow"],
                [0.5, "orange"],
                [1.0, "red"]
            ]
            col1,col2 = st.columns([3,1])
            with col1:
                with st.spinner("Generating map..."):
                    geoScatter(grouped,Status,country,color_scale,center_lon,center_lat)
            with col2:
                st.subheader('')
                Rate(from2,to1,df,country,'Death')
                Rate(from2,to1,df,country,'Recover')
                
                RateMetric(from2,to1,df_rate,country)

            
                
            col1,col2 = st.columns([3,1])  
            with col1:
                st.subheader('Case Status Overview')
                Donut(df_rate,country)
            with col2:
                
                HighestCases(df_rate,country)
                


            
            col1,col2 = st.columns([3,1])
            with col1:
                st.subheader('Top 10 Countries Total_Confirm vs Total_Recovery vs Total_Death')
                fig = px.bar(
                    grouped[grouped['Country/Region'].isin(top_10)][['Country/Region','Confirm','Death','Recover']],
                    x = 'Country/Region',
                    y = ['Confirm','Recover','Death'],
                    # color = y
                    # size = 'Confirm'
                    barmode = 'group'
                )
                st.plotly_chart(fig)
            data = df1[df1['Country/Region'].isin(top_10)][['Country/Region','Recovery_rates']]
            data_new = data.loc[pd.date_range(from2,to1,freq = '35D')].reset_index().rename(columns = {'index': 'Date'}).set_index('Date')
            data_new
            col1,col2 = st.columns([1,3])
            with col2:
                st.subheader('Top 10 Countries Monthly Recovery Rates Area Plot')
                fig = px.area(
                    data_new,
                    x = data_new.index,
                    y = 'Recovery_rates',
                    color = 'Country/Region'
                    
                )
                st.plotly_chart(fig)
            df_rate = df_rate.sort_values('Recover_rate',ascending = False).head(20)
            fig = px.bar(
                df_rate,
                x= 'Country/Region',
                y = 'Recover_rate',
                color = 'Country/Region'
            )
            st.plotly_chart(fig)
            if Status != 'Confirm':
                with col1:
                    data = df1[df1['Confirm']>3000000]
                    data1 = df1[df1['Death']>100000]
                    gr = data.groupby('Country/Region')['Recovery_rates'].mean().sort_values(ascending = False).head(1)
                    gr1 = data1.groupby('Country/Region')['Death_rates'].mean().sort_values(ascending = False).head(1)
                    highest_recovery_rate = gr.index[0]
                    highest_recovery_country = gr.values[0]
                    highest_death_rate = gr1.index[0]
                    highest_death_country = gr1.values[0]
                    
                    
                    # highest_recovery_country = data.sort_values('Recovery_Rate',ascending = False).head(1)['Country/Region'].values[0]
                    # highest_recovery_rate = data.sort_values('Recovery_Rate',ascending = False).head(1)['Recovery_Rate'].values[0]
                    # highest_death_country = data.sort_values('Death_Rate',ascending = False).head(1)['Country/Region'].values[0]
                    # highest_death_rate = data.sort_values('Death_Rate',ascending = False).head(1)['Death_Rate'].values[0]
                    MetricDesign(highest_recovery_rate,highest_recovery_country,'highest Recovery_Rate ')
                    MetricDesign(highest_death_rate,highest_death_country,'highest Death_Rate ')
                

            # grouped['BubbleSize'] = np.log10(grouped['Confirm'] + 1) * 10 + 5
            

        else:   


            df = da
            
            mask = (da.index >= from2) & (da.index <= to1)
            df1 = da[mask]
            mask1 = (monthly_df.index >= from2) & (monthly_df.index <= to1)
            grouped = df1.loc[to1]
            df1 = df1[df1['Country/Region'] == country]
            
            # grouped
            
            # df1
            df1_month = df1.loc[pd.date_range(from2,to1,freq = pd.Timedelta(days = 35))].sort_index(ascending = False)
            # df1_month
            monthly_df1 = monthly_df[mask1]
            # monthly_df1.set_index('Date',inplace = True)
            monthly_df1 = monthly_df1[monthly_df1['Country/Region'] == country]
            new_df = df_confirm_melt.loc[to1]
            df_new = new_df[new_df['Country/Region']==country]
            # df_new
            no_of_province = len(df_new['Province/State'].unique())
            

            
            MetricStruct(from2,to1,df,Status,country)
            col1,col2 = st.columns([3,1])
            color_scale = [
            [0.0, "green"],
            [0.25, "yellow"],
            [0.5, "orange"],
            [1.0, "red"]
            ]
            # grouped['BubbleSize'] = np.log10(grouped['Confirm'] + 1) * 10 + 5
            
            with col1:
                with st.spinner("Generating map..."):
            

                    geoScatter(grouped,Status,country,color_scale,center_lon,center_lat)
            
                
            with col2:
                st.header('')
                # st.header('data')
                # st.write(df1[['Confirm','Single_Confirm']])
                RateMetric(from2,to1,df1,country)
            
            col1,col2 = st.columns([3,1])
            
            
            with col1:
                Confirm = df1.loc[to1]['Confirm']
                Recover = df1.loc[to1]['Recover']
                Death = df1.loc[to1]['Death']
                df_donut = pd.DataFrame({
                    'Category':['Confirm','Recover','Death'],
                    'Values': [Confirm,Recover,Death],
                })
                fig = px.pie(df_donut,names = 'Category',values ='Values',hole = 0.5 )
                st.header('Province wise cases Percentage')
                st.plotly_chart(fig)
            col1,col2 = st.columns([3,1])
            with col1:
                fig = px.line(monthly_df1,
                x = monthly_df1.index,
                y = ['Confirm','Recover','Death'],
                # color = 'Country/Region',
                
                )
                st.header('Daily Line Chart for '+country)
                st.plotly_chart(fig)
            
            
            col1,col2 = st.columns([3,1])
            with col1:
                fig = px.bar(monthly_df1,
                x = monthly_df1.index,
                y = ['Confirm','Recover','Death'],
                )
                st.header('Monthly Bar Chart for '+ country)
                st.plotly_chart(fig)
            col1,col2 = st.columns([3,1])
            with col1:
                fig = px.line(df1,
                x = df1.index,
                y = ['Single_Confirm','Single_Recover','Single_Death'],
                )
                st.header('Monthly Bar Chart for '+ country)
                st.plotly_chart(fig)
            with col2:
                SDay(from1,to,df1,'7Day_Confirm',country)



    else:


        if country == 'World':
            df = da
            mask = (da.index >= from2) & (da.index <= to1)
            df1 = da[mask]
            df_rate = df1[df1['Confirm']>3000000].loc[to1]
            df_rate['Recover_rate'] = round((df_rate['Recover']/df_rate['Confirm'])*100,2)
                # st.write(data.sort_values('Recovery_Rate',ascending = False))
            df_rate['Death_rate'] = round((df_rate['Death']/df_rate['Confirm'])*100,2)
            # df1
            # print(Status)
            MetricStruct(from2,to1,df1,Status,country)
            df2 = df1.loc[to1]
            top_10 = df2.sort_values('Confirm',ascending = False).head(10)['Country/Region'].unique()
            line_df = df[df['Country/Region'].isin(top_10)]
            grouped = df1.loc[to1]
            bar_df = line_df.loc[pd.date_range(from2,to1,freq = '31D')].sort_values('Confirm',ascending = False)
            top_20 = df2.sort_values('Confirm',ascending = False).head(30)['Country/Region'].unique()
            line_df_30 = df[df['Country/Region'].isin(top_20)]

            color_scale = [
                [0.0, "green"],
                [0.25, "yellow"],
                [0.5, "orange"],
                [1.0, "red"]
            ]
            geoScatter(grouped,Status,country,color_scale,center_lon,center_lat)
            # gr_coun_date
            col1,col2 = st.columns([3,1])
            with col1:
                st.subheader('Continent Wise '+Status+' cases Analysis')
                SunPlot(grouped,Status)
            with col2:
                st.header('')
                st.header('')
                g = grouped.groupby('Continent')[Status].sum().sort_values(ascending = False).head(1)
                h = grouped.groupby('Continent')[Status].sum().sort_values().head(1)
                continent_case = g.values[0]
                continent_name = g.index[0]
                MetricDesign(continent_case,continent_name,'Maximum '+Status)
                continent_case1 = h.values[0]
                continent_name1 = h.index[0]
                MetricDesign(continent_case1,continent_name1,'Minimum '+Status)
            st.subheader('Top 10 countries daily '+Status+' cases Line Chart')    
            line_df1 = line_df.reset_index()
            line_df1
            fig = px.line(line_df1,
            x = 'Date',
            y = Status,
            color = 'Country/Region',
            # animation_frame = 'Date',
            # animation_group = 'Country/Region'
            
            )
            st.plotly_chart(fig)
            st.subheader('Top 10 countries daily '+Status+' cases Bar Chart')
            df2 = df1.loc[pd.date_range('2020-1-30','2021-5-29',freq = '7D')]
            


            st.dataframe(df2)
            top_data = {
                "Country": ["USA", "India", "Brazil", "Russia", "UK"],
                "Confirmed": [105_000_000, 45_000_000, 35_000_000, 30_000_000, 25_000_000]
            }
            st.table(top_data)


            fig = px.bar(bar_df,
            x = bar_df.index,
            y = Status,
            color = 'Country/Region',
            # animation_frame = bar_df.index,
            # animation_group = 'Confirm'


            )
            st.plotly_chart(fig)
            if Status != 'Confirm':
                    if Status == 'Recover':
                        col1,col2 = st.columns([2,2])
                        with col1:
                            fig = px.sunburst(df_rate,path= ['Continent','Country/Region'],values = 'Recover_rate',color='Continent',  # use this column to assign color
                                        color_discrete_map={
                                            'Asia': 'lightblue',
                                            'Europe': 'lightgreen'
                                        }) 
                            fig.update_layout(width=500, height=600)
                            fig.update_traces(textfont_size=18)

                            # st.title("🌍 COVID-19 Global Map")
                            st.plotly_chart(fig)
                        with col2:
                            df_rate = df_rate.sort_values('Recover_rate',ascending = False)
                            fig = px.bar(df_rate, x = 'Country/Region',y = 'Recover_rate',color = 'Country/Region')
                            st.plotly_chart(fig)
                    else:
                        col1,col2 = st.columns([2,2])
                        with col1:
                            fig = px.sunburst(df_rate,path= ['Continent','Country/Region'],values = 'Death_rate',color='Continent',  # use this column to assign color
                                        color_discrete_map={
                                            'Asia': 'lightblue',
                                            'Europe': 'lightgreen'
                                        }) 
                            fig.update_layout(width=500, height=600)
                            fig.update_traces(textfont_size=18)

                            # st.title("🌍 COVID-19 Global Map")
                            st.plotly_chart(fig)
                        with col2:
                            df_rate = df_rate.sort_values('Death_rate',ascending = True)
                            fig = px.bar(df_rate, x = 'Country/Region',y = 'Death_rate',color = 'Country/Region')
                            st.plotly_chart(fig)
            
        

        else:   

                df = da
                mask = (da.index >= from2) & (da.index <= to1)
                df1 = da[mask]
                df_confirm_melt['Country/Region'] = df_confirm_melt['Country/Region'].replace('USAA','USA')
                # st.write('HELLO')
                # st.write(df1[df1['Country/Region']=='USA'])
                # df_confirm_melt
                df_rate = df1[df1['Confirm']>3000000].loc[to1]
                df_rate['Recover_rate'] = round((df_rate['Recover']/df_rate['Confirm'])*100,2)
                    # st.write(data.sort_values('Recovery_Rate',ascending = False))
                df_rate['Death_rate'] = round((df_rate['Death']/df_rate['Confirm'])*100,2)
                # df1
                # df_confirm_melt.set_index('Date',inplace = True)
                mask1 = (df_confirm_melt.index >= from2) & (df_confirm_melt.index <= to1)

                mask2 = (df_recover_melt.index >= from2) & (df_recover_melt.index <= to1)

                mask3 = (df_death_melt.index >= from2) & (df_death_melt.index <= to1)
                grouped = df1.loc[to1]
                df1 = df1[df1['Country/Region'] == country]
                # df1
                df1_month = df1.loc[pd.date_range(from2,to1,freq = pd.Timedelta(days = 35))].sort_index(ascending = False)
                # df1_month
                if Status =='Confirm':
                    new_df = df_confirm_melt[mask1]
                    # new_df = new_df['Country/Region'].str.replace('US',"USA")
                    new_df
                    new_df = new_df[new_df['Country/Region'] == country]
                    new_df
                    # x= new_df.groupby('Province/State')['Confirm'].sum()
                    ld = new_df.loc[to1]
                    new_df['Single_Confirm']=new_df.groupby('Province/State')['Confirm'].diff().fillna(0)
                    # x
                    # new_df['Single_confirm'] = new_df[Status].diff()
                else:
                    if Status == 'Recover':
                        new_df = df_recover_melt[mask2]
                        # new_df
                        new_df = new_df[new_df['Country/Region'] == country]
                        # new_df
                        # x= new_df.groupby('Province/State')['Confirm'].sum()
                        ld = new_df.loc[to1]
                        new_df['Single_Recover']=new_df.groupby('Province/State')[Status].diff().fillna(0)
                    else:
                        new_df = df_death_melt[mask3]
                        new_df = new_df[new_df['Country/Region'] == country]
                        # x= new_df.groupby('Province/State')['Confirm'].sum()
                        ld = new_df.loc[to1]
                        new_df['Single_Death']=new_df.groupby('Province/State')[Status].diff().fillna(0)
                new_df
                df_new = new_df[new_df['Country/Region']==country]
                # df_new
                # df_new
                # ld
                
                # new_df
                no_of_province = len(df_new['Province/State'].unique())
                # st.title(Status)

                
                MetricStruct(from2,to1,df1,Status,country)
                col1,col2 = st.columns([3,1])
                color_scale = [
                [0.0, "green"],
                [0.25, "yellow"],
                [0.5, "orange"],
                [1.0, "red"]
                ]
                # grouped['BubbleSize'] = np.log10(grouped['Confirm'] + 1) * 10 + 5
                
                with col1:
                    # (df,status,country,color_scale,center_lon,center_lat)
                    geoScatter(grouped,Status,country,color_scale,center_lon,center_lat)
                with col2:
                    st.header('DataFrame')
                    # st.header('data')
                    st.write(df1[['Country/Region','Confirm','Single_Confirm']])
                col1,col2 = st.columns([3,1])
                with col1:
                    y1 = '7Day_'+Status
                    y2 = 'Single_'+ Status
                    fig = px.line(df1,
                    x = df1.index,
                    y = [y2,y1],
                    # color = y,
                    color_discrete_sequence=['red', 'yellow']
                    


                    )
                    st.header('Daily New Cases vs 7Days Moving Average in '+ country)
                    # fig = px.line(df1,x = df1.index,y = y2)
                    st.plotly_chart(fig)
                if no_of_province>1:
                    col1,col2 = st.columns([3,1])
                    with col1:
                        # ex = [0]*no_of_province
                        # ex[3]= 0.3
                        fig = px.pie(df_new,names = 'Province/State',values = Status,hole = 0.5)
                        st.header('Province wise cases Percentage')
                        st.plotly_chart(fig)
                    with col2:
                        
                        if Status == 'Confirm':
                            max_province_df = ld[ld[Status] == ld[Status].max()]
                            max_province_name = max_province_df['Province/State'].values[0]
                            num = max_province_df[Status].values[0]
                            Single_day_confirm = new_df[new_df['Single_Confirm']==new_df['Single_Confirm'].max()]['Single_Confirm'].values[0]



                            Single_day_confirm_country = new_df[new_df['Single_Confirm']==new_df['Single_Confirm'].max()]['Province/State'].values[0]
                            MetricDesign(num,max_province_name,'Maximum '+ Status)
                            MetricDesign(Single_day_confirm,Single_day_confirm_country,'Single_day Maximum Spike ')
                        if Status == 'Recover':
                            max_province_df = ld[ld[Status] == ld[Status].max()]
                            max_province_name = max_province_df['Province/State'].values[0]
                            num = max_province_df[Status].values[0]
                            Single_day_recover = new_df[new_df['Single_Recover']==new_df['Single_Recover'].max()]['Single_Recover'].values[0]



                            Single_day_recover_country = new_df[new_df['Single_Recover']==new_df['Single_Recover'].max()]['Province/State'].values[0]
                            MetricDesign(num,max_province_name,'Maximum '+ Status)
                            # MetricDesign(Single_day_confirm_country,Single_day_confirm,'Single_day '+ Status)
                            MetricDesign(Single_day_recover,Single_day_recover_country,'Single_day Maximum Spike '+ Status)
                        if Status == 'Death':
                            max_province_df = ld[ld[Status] == ld[Status].max()]
                            max_province_name = max_province_df['Province/State'].values[0]
                            num = max_province_df[Status].values[0]
                            Single_day_death = new_df[new_df['Single_Death']==new_df['Single_Death'].max()]['Single_Death'].values[0]



                            Single_day_death_country = new_df[new_df['Single_Death']==new_df['Single_Death'].max()]['Province/State'].values[0]
                            MetricDesign(num,max_province_name,'Maximum '+ Status)
                            # MetricDesign(Single_day_confirm_country,Single_day_confirm,'Single_day '+ Status)
                            MetricDesign(Single_day_death,Single_day_death_country,'Single_day Maximum Spike '+ Status)
                        
                        
                            


                col1,col2 = st.columns([3,1])
                with col1:
                    fig = px.line(df1,
                    x = df1.index,
                    y = Status,
                    # color = 'Country/Region',
                    
                    )
                    st.header('Daily Line Chart for '+country)
                    st.plotly_chart(fig)
                
                
                col1,col2 = st.columns([3,1])
                with col1:
                    fig = px.bar(df1_month,
                    x = df1_month.index,
                    y = Status,
                    )
                    st.header('Monthly Bar Chart for '+ country)
                    st.plotly_chart(fig)
                
                

summary_input = f"""
You are an expert data analyst. Summarize the following COVID-19 data for the user in clear, concise, and insightful natural language.

User selection:
- Country: {country}
- Date range: {from_str} to {to_str}
- Status: {Status}

Key metrics:
- Total confirmed cases: {total_confirm}
- Total recoveries: {total_recover}
- Total deaths: {total_death}
- Death rate: {death_rate}%

- Highest single-day spike: {max_spike_confirm}



Please provide a short, readable summary highlighting the most important insights, trends, and any anomalies for the selected country and date range.
"""

with tab2:
    if st.button("🧠 AI Generated Summary"):
        with st.spinner("Generating summary..."):
            try:
                response = model.generate_content(
                    f"Write a short, beautiful natural language summary from this COVID-19 data:\n{summary_input}"
                )
                st.success("Here is your summary:")
                st.markdown(f"📢 **AI Summary:**\n\n{response.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # st.text_input()


#AIzaSyAV4-9fv3ltLsPKySpHeRTojdyzr_BXG_o
