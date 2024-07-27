import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Display the icons at the beginning of the app
col1, col2, col3 = st.columns(3)
with col1:
    st.image("icons/github.png", width=32)
    st.markdown('<a href="https://github.com/AdamHassouni" target="_blank">GitHub</a>', unsafe_allow_html=True)
with col2:
    st.image("icons/linkedin.png", width=32)
    st.markdown('<a href="https://www.linkedin.com/in/adam-hassouni/" target="_blank">LinkedIn</a>', unsafe_allow_html=True)
with col3:
    st.image("icons/upwork.png", width=32)
    st.markdown('<a href="https://www.upwork.com/freelancers/~01e711596855735c65" target="_blank">Upwork</a>', unsafe_allow_html=True)


st.title('Black-Scholes Pricing Model')
st.markdown('This app calculates the price of European call and put options using the Black-Scholes formula. The app also includes interactive visualizations of option prices using heatmaps, 3D surface plots, and line plots.')
# Display the main image at the beginning of the app
st.image("icons/image1.jpeg", use_column_width=True)


st.sidebar.header('Parameters')
S = st.sidebar.number_input('Current Asset Price', value=100.0, min_value=0.0, step=1.0)
K = st.sidebar.number_input('Strike Price', value=100.0, min_value=0.0, step=1.0)
T = st.sidebar.number_input('Time to Maturity (Years)', value=1.0, min_value=0.0, step=0.1)
r = st.sidebar.number_input('Risk-Free Interest Rate', value=0.05, min_value=0.0, step=0.01)
sigma = st.sidebar.number_input('Volatility (σ)', value=0.2, min_value=0.0, step=0.01)

st.sidebar.header('Heatmap Parameters')
min_S = st.sidebar.slider('Min Spot Price', min_value=0.0, max_value=200.0, value=80.0)
max_S = st.sidebar.slider('Max Spot Price', min_value=min_S, max_value=200.0, value=120.0)
min_sigma = st.sidebar.slider('Min Volatility ', min_value=0.01, max_value=1.0, value=0.1)
max_sigma = st.sidebar.slider('Max Volatility ', min_value=min_sigma, max_value=1.0, value=0.3)

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

st.markdown("""
    <style>
    .button {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        color: black;
        cursor: pointer;
    }
    .button-call {
        background-color: #90EE90; /* Light green */
    }
    .button-put {
        background-color: #FFCCCC; /* Light red */
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<button class="button button-call">Call Value: {call_price:.2f}</button>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<button class="button button-put">Put Value: {put_price:.2f}</button>', unsafe_allow_html=True)

spot_prices = np.linspace(min_S, max_S, 10)
volatilities = np.linspace(min_sigma, max_sigma, 10)

# Include the initial sigma in the range for the heatmap
volatilities = np.unique(np.append(volatilities, sigma))

call_prices = np.array([[black_scholes_call(S, K, T, r, v) for S in spot_prices] for v in volatilities])
put_prices = np.array([[black_scholes_put(S, K, T, r, v) for S in spot_prices] for v in volatilities])

col1, col2 = st.columns(2)

# Plot Call Price Heatmap in the first column
with col1:
    st.subheader("Call Heatmap")
    fig_call, ax_call = plt.subplots(figsize=(20, 20))
    sns.heatmap(call_prices, ax=ax_call, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), cmap='coolwarm', annot=True, fmt=".2f")
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    st.pyplot(fig_call)

# Plot Put Price Heatmap in the second column
with col2:
    st.subheader("Put Heatmap")
    fig_put, ax_put = plt.subplots(figsize=(20, 20))
    sns.heatmap(put_prices, ax=ax_put, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), cmap='YlGnBu', annot=True, fmt=".2f")
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    st.pyplot(fig_put)

# 3D plot
fig_3d = go.Figure()

fig_3d.add_trace(go.Surface(z=call_prices, x=spot_prices, y=volatilities, colorscale='Viridis', name='Call Prices'))
fig_3d.update_layout(title='3D Surface Plot of Call Prices', scene=dict(
    xaxis_title='Spot Price',
    yaxis_title='Volatility',
    zaxis_title='Call Price'
))

st.plotly_chart(fig_3d)

# Additional Plots

# Line plot of option prices vs. spot price
fig_line_spot = go.Figure()
fig_line_spot.add_trace(go.Scatter(x=spot_prices, y=[black_scholes_call(S, K, T, r, sigma) for S in spot_prices], mode='lines', name='Call Price'))
fig_line_spot.add_trace(go.Scatter(x=spot_prices, y=[black_scholes_put(S, K, T, r, sigma) for S in spot_prices], mode='lines', name='Put Price'))
fig_line_spot.update_layout(title='Option Prices vs. Spot Price', xaxis_title='Spot Price', yaxis_title='Option Price')
st.plotly_chart(fig_line_spot)

# Line plot of option prices vs. volatility
fig_line_volatility = go.Figure()
fig_line_volatility.add_trace(go.Scatter(x=volatilities, y=[black_scholes_call(S, K, T, r, v) for v in volatilities], mode='lines', name='Call Price'))
fig_line_volatility.add_trace(go.Scatter(x=volatilities, y=[black_scholes_put(S, K, T, r, v) for v in volatilities], mode='lines', name='Put Price'))
fig_line_volatility.update_layout(title='Option Prices vs. Volatility', xaxis_title='Volatility', yaxis_title='Option Price')
st.plotly_chart(fig_line_volatility)

# Histogram of call and put prices
call_prices_flat = call_prices.flatten()
put_prices_flat = put_prices.flatten()

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=call_prices_flat, name='Call Prices', opacity=0.75))
fig_hist.add_trace(go.Histogram(x=put_prices_flat, name='Put Prices', opacity=0.75))
fig_hist.update_layout(barmode='overlay', title='Distribution of Option Prices', xaxis_title='Option Price', yaxis_title='Frequency')
fig_hist.update_traces(opacity=0.75)
st.plotly_chart(fig_hist)
