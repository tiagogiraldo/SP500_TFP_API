import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import statsmodels.api as sm
from bokeh.plotting import figure
import matplotlib.pyplot as plt 
import seaborn as sns


today = datetime.date.today()
five_yr_ago = today - datetime.timedelta(days=1828)

st.title("Probabilistic regression model of the Standard & Poor's 500 index stocks")

'''
This app estimates the probabilistic beta of an action belonging to the Standard and 
Poor's 500 index, which is calculated by employing tensorflow 2 and tensorflow 
probability packages.
'''


#sp_500_table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

#df_sp_500 = pd.DataFrame(sp_500_table[0])
path_tickers = "index_ticker/sp500_tickers.csv"
@st.cache(persist=True)
def load_tickers(path_tickers):
    df = pd.read_csv(path_tickers)
    return df

df_sp_500 = load_tickers(path_tickers)
symbol_ls = sorted(df_sp_500['Symbol'].to_list())
sector_ls = sorted(df_sp_500['GICS Sector'].unique())


st.sidebar.markdown("### Select sector")
select_sector = st.sidebar.selectbox("Sector",sector_ls , key ='1')
df_stock_sector = df_sp_500[df_sp_500['GICS Sector']==select_sector] 
stock_sector_list = sorted(df_stock_sector['Symbol'].unique())
select_stock= st.sidebar.selectbox("Sector",stock_sector_list , key ='1')

st.write('The chosen stock is: ', select_stock)

# ----------------------- Data Preprocessing --------------------------------

def download_prices(select_stock, period, progress=False):
#def download_prices(select_stock, start, end, progress=False):
    #df_prices = yf.download(select_stock,start=start, end=end, interval='1mo',progress=False)
    df_prices = yf.download(select_stock,period=period,time_interval='monthly',progress=False)
    return df_prices


selected_period = st.sidebar.selectbox("Select a period among 2 years and 5 years", ["2Y","5Y"],key='1')

#st.sidebar.markdown("**Enter a start and end date.  Experts recommend a range covering 5 years of data for calculating share beta.**")
#start = st.sidebar.text_input('Start date (yyyy-mm-dd)', str(five_yr_ago))
#end = st.sidebar.text_input('End date (yyyy-mm-dd)', str(today))

# downloading selected stock price
stock_df = download_prices(select_stock, selected_period, progress=False)

# Downloadingf SP500 index
sp500_df = download_prices('^GSPC', selected_period, progress=False)

# selecting Adjusted price column
sp500_df_Adj_close = sp500_df[['Adj Close']]
stock_df_Adj_close = stock_df[['Adj Close']]

# Getting the frequency method
selected_freq = st.sidebar.selectbox('Frequency return method (Weekly: W-FRI , Monthly: M)', ['W-FRI', 'M'], key='1')


# Getting frequency returns
frequency_sp500_returns = sp500_df_Adj_close.pct_change(periods = 1, freq = selected_freq).dropna(how='all')
frequency_stock_returns = stock_df_Adj_close.pct_change(periods = 1, freq = selected_freq).dropna(how='all')

# Renaming df column names
frequency_sp500_returns.columns=['SP500_return']
frequency_stock_returns.columns=[str(select_stock)+'_return']


# Joinin both df
return_df = pd.concat([frequency_sp500_returns, frequency_stock_returns], axis=1, sort=False)

# Drop the first porcentage calculation

# Because the way pandas compute the first percentage, it should be necessary drop the first 
# calculation, It takes the first time series value (no necessarily the month first register 
# because the way it was downloaded) and end of the month value to compute the first monthly 
# return.  It is an imprecision, and the first value calculated must be removed from dataframe

return_df = return_df.iloc[1:]
st.write(return_df)

# ----------------- No Uncertainty Model --------------------------
# Regression computations
Y = return_df[str(select_stock)+'_return']
X = return_df['SP500_return']
X = sm.add_constant(X)
mod = sm.OLS(Y,X)
res = mod.fit()

# Creating results pandas dataframe
res_df = pd.read_html(res.summary().tables[1].as_html(),header=0,index_col=0)[0]

intercept = res_df['coef'].values[0]
slope=res_df['coef'].values[1]

# Generating predicted data
x_ls = return_df['SP500_return'].to_list()
y_predicted = [slope * i + intercept  for i in x_ls]
return_df['y_predicted'] = y_predicted


# Plotting scatter data and regression line
st.subheader('Linear Regression With No Uncertainty Scatter Plot')
p = figure(
     title='Scatter plot SP500 and ' + str(select_stock) + ' returns',
     x_axis_label= 'SP500 return',
     y_axis_label=str(select_stock) + ' return'
     )

p.scatter('SP500_return',str(select_stock)+'_return',marker="square", fill_color="red", source = return_df)
p.line('SP500_return','y_predicted', color='blue', source = return_df, line_width=2)
st.bokeh_chart(p, use_container_width=True)

average_return = return_df.mean(axis=0)
#average_return
loc = [average_return[0], average_return[1]]
#loc
std_return = return_df.std(axis=0)
#scale = [std_return[0], std_return[1] ]
#scale

st.subheader('Linear regression With No Uncertainty')

st.text(res.summary())

# --------------- Probabilistic Model ---------------------------

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions

if tf.test.gpu_device_name() != '/device:GPU:0':
  st.write('WARNING: GPU device not found.')
else:
  st.write('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))


# Defining negative log-likelihood
nll = lambda y, rv_y: -rv_y.log_prob(y)

# Passing df data to numpy arras

x = return_df['SP500_return'].to_numpy(dtype=np.float32)
y = return_df[str(select_stock)+'_return'].to_numpy(dtype=np.float32)

# Aleatory Uncertainty
# Build model.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1 + 1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=nll)
model.fit(x, y, epochs=1000, verbose=False);

# Profit.
yhat = model.predict(x)



# https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression


w = np.squeeze(model.layers[-2].kernel.numpy())
b = np.squeeze(model.layers[-2].bias.numpy())


y_hat = yhat.tolist()
y_hat_ls = [y_hat[i][0] for i in range(yhat.shape[0])]

merge_ret_df = pd.DataFrame({'SP500_return': list(x), 'y_hat': y_hat_ls})
#merge_ret_df

st.subheader('Linear Regression With Uncertainty Scatter Plot')
pu = figure(
     title='Regression With Epistemic Uncertainty',
     x_axis_label= 'SP500 return',
     y_axis_label=str(select_stock) + 'return'
     )
pu.scatter('SP500_return', str(select_stock)+'_return',marker="square", fill_color="red", source = return_df)
pu.scatter('SP500_return', 'y_hat', marker="circle",fill_color='green', line_width=2, source = merge_ret_df)
st.bokeh_chart(pu, use_container_width=True)























st.subheader('Epistemic vs. Aleatory uncertainty \n\n') 
st.markdown('Uncertainty is categorized into two types: epistemic (also known as systematic or reducible \
uncertainty) and aleatory (also known as statistical or irreducible uncertainty). \n') 
st.markdown('**Epistemic Uncertainty** derives its name from the Greek word “επιστήμη” (episteme) \
which can be roughly translated as knowledge. Therefore, epistemic uncertainty is presumed to derive from the lack of knowledge of information regarding the phenomena \
that dictate how a system should behave, ultimately affecting the outcome of an event. \n') 

st.markdown('**Aleatory Uncertainty** derives its name from the Latin word “alea” which is \
translated as “the roll of the dice”. Therefore, aleatory uncertainty can be defined as the internal randomness of phenomena [2]. ')

st.markdown('[Source](http://apppm.man.dtu.dk/index.php/Epistemic_vs._Aleatory_uncertainty)')



