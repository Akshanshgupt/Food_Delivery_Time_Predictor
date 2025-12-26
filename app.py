import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Paths
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "Food_Delivery_Times.csv"
MODEL_PATH = BASE_DIR / "rf_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    df = pd.read_csv(CSV_PATH)

    if 'Order_ID' in df.columns:
        df = df.drop(columns=['Order_ID'])

    df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)
    df['Weather'].fillna(df['Weather'].mode()[0], inplace=True)
    df['Traffic_Level'].fillna(df['Traffic_Level'].mode()[0], inplace=True)
    df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0], inplace=True)
    df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0], inplace=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    numeric_df = df.select_dtypes(include='number')
    describe = numeric_df.describe()

    return model, feature_columns, encoders, df, describe


def encode_input(user_df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df_enc = user_df.copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        df_enc[col] = df_enc[col].apply(lambda x: x if x in known else None)
        most_freq = le.classes_[0] if len(le.classes_) else None
        df_enc[col] = df_enc[col].fillna(most_freq)
        df_enc[col] = le.transform(df_enc[col])
    return df_enc


def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 60], 'tickwidth': 1},
            'bar': {'color': "#FF4B4B"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#90EE90'},
                {'range': [20, 35], 'color': '#FFD700'},
                {'range': [35, 60], 'color': '#FFB6C1'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🚚", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #FF4B4B;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-size: 1.2rem;
            padding: 0.75rem;
            border-radius: 10px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #E03E3E;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🚚 Food Delivery Time Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Random Forest ML Model | Accuracy: 79%</div>', unsafe_allow_html=True)

    try:
        model, feature_columns, encoders, train_df, desc = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

    # Initialize session state for prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.metric("Model Type", "Random Forest")
    st.sidebar.metric("R² Score", "0.79")
    st.sidebar.metric("Training Samples", len(train_df))
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Quick Stats")
    st.sidebar.metric("Avg Delivery Time", f"{train_df['Delivery_Time_min'].mean():.1f} min")
    st.sidebar.metric("Min Delivery Time", f"{train_df['Delivery_Time_min'].min():.0f} min")
    st.sidebar.metric("Max Delivery Time", f"{train_df['Delivery_Time_min'].max():.0f} min")

    # Main input section
    st.markdown("### 📝 Enter Delivery Details")
    
    def rng(col, step=0.01):
        try:
            cmin = float(desc.loc['min', col])
            cmax = float(desc.loc['max', col])
            cmean = float(desc.loc['mean', col])
            return cmin, cmax, cmean
        except Exception:
            series = train_df[col]
            return float(series.min()), float(series.max()), float(series.mean())

    col1, col2, col3 = st.columns(3)

    with col1:
        dmin, dmax, dmean = rng('Distance_km')
        distance_km = st.slider(
            '📍 Distance (km)',
            min_value=round(dmin, 2), 
            max_value=round(dmax, 2), 
            value=round(dmean, 2), 
            step=0.1
        )

    with col2:
        pmin, pmax, pmean = rng('Preparation_Time_min')
        prep_time = st.slider(
            '🍳 Preparation Time (min)',
            min_value=int(pmin), 
            max_value=int(pmax), 
            value=int(round(pmean)), 
            step=1
        )

    with col3:
        emin, emax, emean = rng('Courier_Experience_yrs')
        exp_years = st.slider(
            '👤 Courier Experience (yrs)',
            min_value=float(emin), 
            max_value=float(emax), 
            value=float(round(emean, 1)), 
            step=0.5
        )

    col4, col5, col6 = st.columns(3)
    
    with col4:
        vehicle_options = sorted(train_df['Vehicle_Type'].unique().tolist())
        vehicle = st.selectbox('🏍️ Vehicle Type', vehicle_options, help="Select the delivery vehicle type")

    with col5:
        traffic_options = sorted(train_df['Traffic_Level'].unique().tolist())
        traffic = st.selectbox('🚦 Traffic Level', traffic_options, help="Current traffic conditions")
    
    with col6:
        weather_options = sorted(train_df['Weather'].unique().tolist())
        weather = st.selectbox('🌤️ Weather', weather_options, help="Current weather conditions")

    time_options = sorted(train_df['Time_of_Day'].unique().tolist())
    time_of_day = st.selectbox('🕐 Time of Day', time_options, help="When is the delivery scheduled?")

    st.markdown("---")
    
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button('🎯 PREDICT DELIVERY TIME', use_container_width=True)

    if predict_button:
        input_dict = {
            'Distance_km': [distance_km],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [exp_years],
        }
        user_df = pd.DataFrame(input_dict)

        user_df['Courier_Experience_yrs'].fillna(train_df['Courier_Experience_yrs'].median(), inplace=True)
        for col in ['Weather','Traffic_Level','Time_of_Day','Vehicle_Type']:
            user_df[col].fillna(train_df[col].mode()[0], inplace=True)

        user_encoded = encode_input(user_df, encoders)
        X_input = user_encoded.reindex(columns=feature_columns, fill_value=0)

        try:
            pred = model.predict(X_input)[0]
            
            # Add to history
            st.session_state.prediction_history.append({
                'distance': distance_km,
                'prep_time': prep_time,
                'prediction': pred
            })
            
            # Keep only last 10 predictions
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history.pop(0)
            
            st.markdown("---")
            st.markdown("### 🎉 Prediction Result")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.plotly_chart(create_gauge_chart(pred, "Estimated Delivery Time (min)"), use_container_width=True)
            
            with result_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if pred < 20:
                    status = "🟢 Fast Delivery"
                    message = "Great! This delivery should be quick."
                elif pred < 35:
                    status = "🟡 Average Delivery"
                    message = "Standard delivery time expected."
                else:
                    status = "🔴 Slow Delivery"
                    message = "This might take a while. Consider informing the customer."
                
                st.markdown(f"### {status}")
                st.markdown(f"**{message}**")
                st.metric("Estimated Time", f"{pred:.1f} minutes", delta=f"{pred - train_df['Delivery_Time_min'].mean():.1f} min vs avg")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Feature Impact Visualization
    with st.expander("📊 View Prediction History & Insights", expanded=False):
        if len(st.session_state.prediction_history) > 0:
            st.markdown("#### Recent Predictions")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            fig = px.line(history_df, y='prediction', 
                         title='Your Recent Prediction Trends',
                         labels={'index': 'Prediction #', 'prediction': 'Time (min)'},
                         markers=True)
            fig.update_traces(line_color='#FF4B4B', line_width=3, marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.scatter(history_df, x='distance', y='prediction', 
                                 size='prep_time', title='Distance vs Delivery Time',
                                 labels={'distance': 'Distance (km)', 'prediction': 'Time (min)'})
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.bar(history_df.tail(5), x=history_df.tail(5).index, y='prediction',
                             title='Last 5 Predictions',
                             labels={'index': 'Prediction #', 'prediction': 'Time (min)'})
                fig3.update_traces(marker_color='#FF4B4B')
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Make your first prediction to see insights here!")

    # Data Distribution Insights
    with st.expander("📈 Dataset Statistics & Distribution", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(train_df, x='Delivery_Time_min', 
                                    title='Distribution of Delivery Times',
                                    labels={'Delivery_Time_min': 'Delivery Time (min)'},
                                    color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_vehicle = px.pie(train_df, names='Vehicle_Type', 
                                title='Vehicle Type Distribution',
                                color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_vehicle, use_container_width=True)


if __name__ == '__main__':
    main()