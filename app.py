import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🌤️ Weather Forecast Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5em;
        margin: 0;
    }
    .main-header p {
        color: #a0c4ff;
        font-size: 1.1em;
        margin-top: 5px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    .metric-card h3 {
        color: #a0c4ff;
        font-size: 0.9em;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        color: #ffffff;
        font-size: 2em;
        margin: 5px 0;
    }
    
    /* Forecast card */
    .forecast-card {
        background: linear-gradient(135deg, #0f3460, #533483);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .forecast-card:hover {
        transform: translateY(-5px);
    }
    .forecast-card .day {
        color: #a0c4ff;
        font-size: 0.85em;
        font-weight: bold;
    }
    .forecast-card .temp {
        color: #ffffff;
        font-size: 1.8em;
        font-weight: bold;
        margin: 5px 0;
    }
    .forecast-card .date {
        color: #c4c4c4;
        font-size: 0.75em;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #134e5e, #71b280);
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e3c72;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained ARIMA model and artifacts"""
    try:
        with open('weather_arima_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts, True
    except FileNotFoundError:
        return None, False

@st.cache_data
def load_historical_data():
    """Load historical weather data"""
    try:
        with open('historical_weather_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data, True
    except FileNotFoundError:
        return None, False

def get_weather_emoji(temp):
    """Return weather emoji based on temperature"""
    if temp < 0:
        return "🥶"
    elif temp < 10:
        return "❄️"
    elif temp < 20:
        return "🌤️"
    elif temp < 30:
        return "☀️"
    elif temp < 35:
        return "🔥"
    else:
        return "🌡️"

def get_temp_color(temp):
    """Return color based on temperature"""
    if temp < 0:
        return "#00bfff"
    elif temp < 10:
        return "#4fc3f7"
    elif temp < 20:
        return "#81c784"
    elif temp < 25:
        return "#ffb74d"
    elif temp < 30:
        return "#ff8a65"
    else:
        return "#ef5350"

def make_forecast(model, steps=7):
    """Generate forecast with confidence intervals"""
    forecast_result = model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)
    return forecast_result

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌤️ Weather Forecast Dashboard</h1>
        <p>ARIMA Time Series Model | Next 7 Days Temperature Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    artifacts, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ **Model not found!** Please run `trainmodel.py` first to train and save the model.")
        st.code("python trainmodel.py", language="bash")
        st.info("This will download the weather dataset, train an ARIMA model, and save it as a pickle file.")
        
        st.markdown("### 📋 Steps to Get Started:")
        st.markdown("""
        1. Install requirements: `pip install kagglehub statsmodels pandas numpy scikit-learn`
        2. Run training: `python trainmodel.py`
        3. Run dashboard: `streamlit run app.py`
        """)
        return
    
    # Extract artifacts
    model = artifacts['model']
    order = artifacts['order']
    metrics = artifacts['metrics']
    historical_data = artifacts['historical_data']
    additional_features = artifacts.get('additional_features', pd.DataFrame())
    training_info = artifacts['training_info']
    last_date = artifacts['last_date']
    
    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown("---")
        
        # Forecast days slider
        forecast_days = st.slider(
            "📅 Forecast Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to forecast ahead"
        )
        
        # Historical days to show
        hist_days = st.slider(
            "📊 Historical Days to Display",
            min_value=7,
            max_value=365,
            value=90,
            help="Number of historical days to show in charts"
        )
        
        # Confidence interval
        show_ci = st.checkbox("📐 Show Confidence Interval", value=True)
        
        # Chart theme
        chart_theme = st.selectbox(
            "🎨 Chart Theme",
            ["plotly_dark", "plotly", "plotly_white", "ggplot2", "seaborn"],
            index=0
        )
        
        st.markdown("---")
        
        # Model Information
        st.markdown("## 📊 Model Info")
        st.markdown(f"""
        - **Model:** ARIMA{order}
        - **AIC:** {training_info['model_aic']:.2f}
        - **Training Data:** {training_info['total_observations']} days
        - **Period:** {training_info['date_range_start'][:10]} to {training_info['date_range_end'][:10]}
        """)
        
        st.markdown("---")
        st.markdown("## 📏 Test Metrics")
        st.metric("MAE", f"{metrics['mae']:.2f} °C")
        st.metric("RMSE", f"{metrics['rmse']:.2f} °C")
        st.metric("MAPE", f"{metrics['mape']:.2f}%")
    
    # ============================================================
    # GENERATE FORECAST
    # ============================================================
    forecast_result = make_forecast(model, steps=forecast_days)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)
    
    # Create forecast DataFrame using model's forecast index
    forecast_df = pd.DataFrame({
        'Date': forecast_mean.index,
        'Temperature': forecast_mean.values,
        'Lower_CI': forecast_ci.iloc[:, 0].values,
        'Upper_CI': forecast_ci.iloc[:, 1].values
    })
    forecast_dates = forecast_df['Date']
    forecast_df['Day'] = forecast_df['Date'].dt.strftime('%A')
    forecast_df['Date_Str'] = forecast_df['Date'].dt.strftime('%b %d, %Y')
    
    # ============================================================
    # TOP METRICS ROW
    # ============================================================
    st.markdown("### 📈 Quick Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_temp = historical_data['Temperature'].iloc[-1]
    avg_forecast = forecast_df['Temperature'].mean()
    max_forecast = forecast_df['Temperature'].max()
    min_forecast = forecast_df['Temperature'].min()
    temp_trend = forecast_df['Temperature'].iloc[-1] - forecast_df['Temperature'].iloc[0]
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Last Recorded</h3>
            <h2>{current_temp:.1f}°C {get_weather_emoji(current_temp)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Forecast</h3>
            <h2>{avg_forecast:.1f}°C {get_weather_emoji(avg_forecast)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Max Forecast</h3>
            <h2>{max_forecast:.1f}°C {get_weather_emoji(max_forecast)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Min Forecast</h3>
            <h2>{min_forecast:.1f}°C {get_weather_emoji(min_forecast)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        trend_arrow = "↑" if temp_trend > 0 else "↓" if temp_trend < 0 else "→"
        trend_color = "#ef5350" if temp_trend > 0 else "#4fc3f7" if temp_trend < 0 else "#ffffff"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Trend ({forecast_days}d)</h3>
            <h2 style="color: {trend_color};">{trend_arrow} {abs(temp_trend):.1f}°C</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # FORECAST CARDS (Next 7 days)
    # ============================================================
    st.markdown(f"### 🗓️ Next {min(forecast_days, 7)} Days Forecast")
    
    display_days = min(forecast_days, 7)
    cols = st.columns(display_days)
    
    for i, col in enumerate(cols):
        if i < len(forecast_df):
            row = forecast_df.iloc[i]
            emoji = get_weather_emoji(row['Temperature'])
            with col:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="day">{row['Day'][:3].upper()}</div>
                    <div style="font-size: 2em;">{emoji}</div>
                    <div class="temp">{row['Temperature']:.1f}°C</div>
                    <div class="date">{row['Date_Str']}</div>
                    <div style="color: #a0a0a0; font-size: 0.7em; margin-top: 5px;">
                        {row['Lower_CI']:.1f}° / {row['Upper_CI']:.1f}°
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # TABS FOR DIFFERENT VIEWS
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast Chart", 
        "📊 Historical Analysis", 
        "🔍 Model Evaluation",
        "📋 Data Tables"
    ])
    
    # ============================================================
    # TAB 1: FORECAST CHART
    # ============================================================
    with tab1:
        st.markdown("### 🔮 Temperature Forecast with Historical Context")
        
        # Get historical data for display
        hist_display = historical_data.iloc[-hist_days:]
        
        # Create the main forecast chart
        fig = go.Figure()
        
        # Historical temperature line
        fig.add_trace(go.Scatter(
            x=hist_display.index,
            y=hist_display['Temperature'],
            mode='lines',
            name='Historical Temperature',
            line=dict(color='#4fc3f7', width=1.5),
            opacity=0.8,
            hovertemplate='Date: %{x|%b %d, %Y}<br>Temperature: %{y:.1f}°C<extra></extra>'
        ))
        
        # Historical moving average
        ma_window = min(30, len(hist_display) // 3)
        if ma_window > 1:
            hist_ma = hist_display['Temperature'].rolling(window=ma_window).mean()
            fig.add_trace(go.Scatter(
                x=hist_display.index,
                y=hist_ma,
                mode='lines',
                name=f'{ma_window}-Day Moving Average',
                line=dict(color='#ffb74d', width=2, dash='dot'),
                opacity=0.7,
                hovertemplate='Date: %{x|%b %d, %Y}<br>MA: %{y:.1f}°C<extra></extra>'
            ))
        
        # Confidence interval (shaded area)
        if show_ci:
            fig.add_trace(go.Scatter(
                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                y=pd.concat([forecast_df['Upper_CI'], forecast_df['Lower_CI'][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 107, 107, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df['Temperature'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=10, color='#ff6b6b', 
                       line=dict(width=2, color='white'),
                       symbol='diamond'),
            hovertemplate='Date: %{x|%b %d, %Y}<br>Forecast: %{y:.1f}°C<extra></extra>'
        ))
        
        # Add vertical line at forecast start
        fig.add_shape(
            type="line",
            x0=last_date,
            x1=last_date,
            y0=0,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color="yellow", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=last_date,
            y=1,
            xref='x',
            yref='paper',
            text="Forecast Start",
            showarrow=False,
            font=dict(color="yellow"),
            yanchor="bottom"
        )
        
        fig.update_layout(
            template=chart_theme,
            title=dict(
                text=f'Temperature: Last {hist_days} Days + {forecast_days}-Day Forecast',
                font=dict(size=20)
            ),
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            height=550,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            margin=dict(l=60, r=30, t=80, b=60)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Forecast only zoom chart
        st.markdown("### 🔍 Forecast Detail View")
        
        fig_detail = go.Figure()
        
        # Confidence interval
        if show_ci:
            fig_detail.add_trace(go.Scatter(
                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                y=pd.concat([forecast_df['Upper_CI'], forecast_df['Lower_CI'][::-1]]),
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
            ))
        
        # Forecast bars
        colors = [get_temp_color(t) for t in forecast_df['Temperature']]
        
        fig_detail.add_trace(go.Bar(
            x=forecast_dates,
            y=forecast_df['Temperature'],
            name='Forecast Temperature',
            marker_color=colors,
            opacity=0.8,
            text=[f"{t:.1f}°C" for t in forecast_df['Temperature']],
            textposition='outside',
            textfont=dict(color='white', size=12),
            hovertemplate='Date: %{x|%A, %b %d}<br>Temperature: %{y:.1f}°C<extra></extra>'
        ))
        
        # Forecast line overlay
        fig_detail.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df['Temperature'],
            mode='lines+markers',
            name='Trend Line',
            line=dict(color='white', width=2, dash='dot'),
            marker=dict(size=8, color='white'),
            hoverinfo='skip'
        ))
        
        fig_detail.update_layout(
            template=chart_theme,
            title=f'{forecast_days}-Day Temperature Forecast',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            height=450,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
            bargap=0.3
        )
        
        st.plotly_chart(fig_detail, width='stretch')
    
    # ============================================================
    # TAB 2: HISTORICAL ANALYSIS
    # ============================================================
    with tab2:
        st.markdown("### 📊 Historical Temperature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=historical_data['Temperature'],
                nbinsx=50,
                name='Temperature Distribution',
                marker_color='#4fc3f7',
                opacity=0.7
            ))
            fig_dist.add_vline(
                x=historical_data['Temperature'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {historical_data['Temperature'].mean():.1f}°C"
            )
            fig_dist.update_layout(
                template=chart_theme,
                title='Temperature Distribution',
                xaxis_title='Temperature (°C)',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig_dist, width='stretch')
        
        with col2:
            # Monthly average temperature
            monthly_data = historical_data.copy()
            monthly_data['Month'] = monthly_data.index.month
            monthly_data['Month_Name'] = monthly_data.index.strftime('%B')
            monthly_avg = monthly_data.groupby('Month')['Temperature'].agg(['mean', 'std', 'min', 'max']).reset_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg['Month_Name'] = [month_names[m-1] for m in monthly_avg['Month']]
            
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_avg['Month_Name'],
                y=monthly_avg['mean'],
                name='Avg Temperature',
                marker_color=px.colors.sequential.Viridis[:len(monthly_avg)],
                error_y=dict(type='data', array=monthly_avg['std'], visible=True),
                text=[f"{t:.1f}°C" for t in monthly_avg['mean']],
                textposition='outside'
            ))
            fig_monthly.update_layout(
                template=chart_theme,
                title='Monthly Average Temperature',
                xaxis_title='Month',
                yaxis_title='Temperature (°C)',
                height=400
            )
            st.plotly_chart(fig_monthly, width='stretch')
        
        # Year-over-year comparison
        st.markdown("### 📅 Yearly Temperature Trends")
        yearly_data = historical_data.copy()
        yearly_data['Year'] = yearly_data.index.year
        yearly_data['DayOfYear'] = yearly_data.index.dayofyear
        
        fig_yearly = go.Figure()
        years = sorted(yearly_data['Year'].unique())
        colors = px.colors.qualitative.Set2
        
        for i, year in enumerate(years):
            year_data = yearly_data[yearly_data['Year'] == year]
            # Smooth with rolling average
            smoothed = year_data['Temperature'].rolling(window=7, center=True).mean()
            fig_yearly.add_trace(go.Scatter(
                x=year_data['DayOfYear'],
                y=smoothed,
                mode='lines',
                name=str(year),
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.7
            ))
        
        fig_yearly.update_layout(
            template=chart_theme,
            title='Temperature by Day of Year (7-day smoothed)',
            xaxis_title='Day of Year',
            yaxis_title='Temperature (°C)',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_yearly, width='stretch')
        
        # Additional features (if available)
        if len(additional_features.columns) > 0:
            st.markdown("### 🌡️ Additional Weather Metrics")
            
            # Show last hist_days of additional features
            add_feat_display = additional_features.iloc[-hist_days:]
            
            n_features = len(add_feat_display.columns)
            if n_features > 0:
                fig_features = make_subplots(
                    rows=(n_features + 1) // 2, cols=2,
                    subplot_titles=add_feat_display.columns.tolist(),
                    vertical_spacing=0.1
                )
                
                feature_colors = ['#4fc3f7', '#81c784', '#ffb74d', '#ff8a65', '#ba68c8']
                
                for idx, col_name in enumerate(add_feat_display.columns):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    fig_features.add_trace(
                        go.Scatter(
                            x=add_feat_display.index,
                            y=add_feat_display[col_name],
                            mode='lines',
                            name=col_name,
                            line=dict(color=feature_colors[idx % len(feature_colors)], width=1),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig_features.update_layout(
                    template=chart_theme,
                    height=300 * ((n_features + 1) // 2),
                    title_text="Additional Weather Features"
                )
                st.plotly_chart(fig_features, width='stretch')
    
    # ============================================================
    # TAB 3: MODEL EVALUATION
    # ============================================================
    with tab3:
        st.markdown("### 🔍 ARIMA Model Evaluation")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Mean Absolute Error</h3>
                <h2>{metrics['mae']:.4f}°C</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Root Mean Squared Error</h3>
                <h2>{metrics['rmse']:.4f}°C</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Mean Abs % Error</h3>
                <h2>{metrics['mape']:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Actual vs Predicted on test set
        st.markdown("### 📉 Actual vs Predicted (Test Set)")
        
        test_dates = metrics['test_dates']
        test_actual = metrics['test_actual']
        test_predicted = metrics['test_predicted']
        
        fig_eval = go.Figure()
        
        fig_eval.add_trace(go.Scatter(
            x=test_dates,
            y=test_actual,
            mode='lines',
            name='Actual Temperature',
            line=dict(color='#4fc3f7', width=2),
            hovertemplate='Date: %{x|%b %d}<br>Actual: %{y:.1f}°C<extra></extra>'
        ))
        
        fig_eval.add_trace(go.Scatter(
            x=test_dates,
            y=test_predicted,
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            hovertemplate='Date: %{x|%b %d}<br>Predicted: %{y:.1f}°C<extra></extra>'
        ))
        
        fig_eval.update_layout(
            template=chart_theme,
            title='Test Set: Actual vs Predicted Temperature',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_eval, width='stretch')
        
        # Residuals analysis
        col1, col2 = st.columns(2)
        
        residuals = np.array(test_actual) - np.array(test_predicted)
        
        with col1:
            # Residual distribution
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residuals',
                marker_color='#ba68c8',
                opacity=0.7
            ))
            fig_resid.add_vline(x=0, line_dash="dash", line_color="white")
            fig_resid.add_vline(
                x=np.mean(residuals),
                line_dash="dot",
                line_color="red",
                annotation_text=f"Mean: {np.mean(residuals):.2f}"
            )
            fig_resid.update_layout(
                template=chart_theme,
                title='Residual Distribution',
                xaxis_title='Residual (°C)',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig_resid, width='stretch')
        
        with col2:
            # Residuals over time
            fig_resid_time = go.Figure()
            fig_resid_time.add_trace(go.Scatter(
                x=test_dates,
                y=residuals,
                mode='markers+lines',
                name='Residuals',
                marker=dict(
                    color=residuals,
                    colorscale='RdBu',
                    size=6,
                    showscale=True,
                    colorbar=dict(title='Residual')
                ),
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ))
            fig_resid_time.add_hline(y=0, line_dash="dash", line_color="yellow")
            fig_resid_time.update_layout(
                template=chart_theme,
                title='Residuals Over Time',
                xaxis_title='Date',
                yaxis_title='Residual (°C)',
                height=400
            )
            st.plotly_chart(fig_resid_time, width='stretch')
        
        # Error scatter plot
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=test_actual,
            y=test_predicted,
            mode='markers',
            name='Predictions',
            marker=dict(color='#4fc3f7', size=6, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(min(test_actual), min(test_predicted))
        max_val = max(max(test_actual), max(test_predicted))
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_scatter.update_layout(
            template=chart_theme,
            title='Actual vs Predicted Scatter Plot',
            xaxis_title='Actual Temperature (°C)',
            yaxis_title='Predicted Temperature (°C)',
            height=450
        )
        
        st.plotly_chart(fig_scatter, width='stretch')
        
        # Model info box
        st.markdown(f"""
        <div class="info-box">
            <h4>📋 Model Configuration</h4>
            <p><strong>Model Type:</strong> ARIMA (AutoRegressive Integrated Moving Average)</p>
            <p><strong>Order (p,d,q):</strong> {order}</p>
            <p><strong>p (AR terms):</strong> {order[0]} — Number of autoregressive terms</p>
            <p><strong>d (Differencing):</strong> {order[1]} — Number of times the series is differenced</p>
            <p><strong>q (MA terms):</strong> {order[2]} — Number of moving average terms</p>
            <p><strong>AIC Score:</strong> {training_info['model_aic']:.2f}</p>
            <p><strong>Training Period:</strong> {training_info['date_range_start'][:10]} to {training_info['date_range_end'][:10]}</p>
            <p><strong>Total Observations:</strong> {training_info['total_observations']} days</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # TAB 4: DATA TABLES
    # ============================================================
    with tab4:
        st.markdown("### 📋 Forecast Data Table")
        
        # Forecast table with formatting
        display_forecast = forecast_df[['Date_Str', 'Day', 'Temperature', 'Lower_CI', 'Upper_CI']].copy()
        display_forecast.columns = ['Date', 'Day', 'Temperature (°C)', 'Lower 95% CI', 'Upper 95% CI']
        display_forecast['Temperature (°C)'] = display_forecast['Temperature (°C)'].round(2)
        display_forecast['Lower 95% CI'] = display_forecast['Lower 95% CI'].round(2)
        display_forecast['Upper 95% CI'] = display_forecast['Upper 95% CI'].round(2)
        display_forecast['Weather'] = [get_weather_emoji(t) for t in forecast_df['Temperature']]
        display_forecast.index = range(1, len(display_forecast) + 1)
        display_forecast.index.name = 'Day #'
        
        st.dataframe(
            display_forecast,
            width='stretch',
            height=min(400, len(display_forecast) * 40 + 50)
        )
        
        # Download button for forecast
        csv = display_forecast.to_csv(index=True)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name=f"weather_forecast_{forecast_days}days.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        
        # Historical data summary
        st.markdown("### 📊 Historical Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Last 30 Days")
            last_30 = historical_data.iloc[-30:].copy()
            last_30['Date'] = last_30.index.strftime('%b %d, %Y')
            last_30['Day'] = last_30.index.strftime('%A')
            last_30 = last_30[['Date', 'Day', 'Temperature']].reset_index(drop=True)
            last_30['Temperature'] = last_30['Temperature'].round(2)
            last_30.index = range(1, len(last_30) + 1)
            st.dataframe(last_30, width='stretch', height=400)
        
        with col2:
            st.markdown("#### Statistical Summary")
            stats = historical_data['Temperature'].describe()
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 
                             'Median', '75th Percentile', 'Max'],
                'Value': [
                    f"{stats['count']:.0f} days",
                    f"{stats['mean']:.2f} °C",
                    f"{stats['std']:.2f} °C",
                    f"{stats['min']:.2f} °C",
                    f"{stats['25%']:.2f} °C",
                    f"{stats['50%']:.2f} °C",
                    f"{stats['75%']:.2f} °C",
                    f"{stats['max']:.2f} °C"
                ]
            })
            stats_df.index = range(1, len(stats_df) + 1)
            st.dataframe(stats_df, width='stretch', height=350)
            
            # Download historical data
            hist_csv = historical_data.to_csv()
            st.download_button(
                label="📥 Download Historical Data",
                data=hist_csv,
                file_name="historical_weather_data.csv",
                mime="text/csv",
            )
    
    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>🌤️ <strong>Weather Forecast Dashboard</strong> | Built with ARIMA & Streamlit</p>
        <p style="font-size: 0.8em;">
            Model: ARIMA{order} | Dataset: Muthuj7 Weather Dataset (Kaggle) | 
            Last Updated: {date}
        </p>
    </div>
    """.format(order=order, date=datetime.now().strftime('%B %d, %Y')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()