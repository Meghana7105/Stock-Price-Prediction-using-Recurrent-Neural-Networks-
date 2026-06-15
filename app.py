import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Stock Price Prediction System")

st.title("📈 Stock Price Prediction System (RNN)")
st.markdown("---")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load Data with error handling
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Convert Date properly
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
        data = data.sort_values("Date")  # Ensure chronological order
        
        # Remove rows with invalid dates
        data = data.dropna(subset=['Date'])
    
    # -----------------------------
    # AUTO CLEAN DATA (No warnings shown)
    # -----------------------------
    if "Close" not in data.columns:
        st.error("CSV must contain a 'Close' column.")
        st.stop()
    
    # Convert Close to numeric, coercing errors to NaN
    data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
    
    # AUTOMATICALLY remove rows with NaN in Close column
    data = data.dropna(subset=['Close'])
    
    # OPTION: Remove future dates (uncomment if you want to remove future dates)
    # current_date = pd.Timestamp.now()
    # data = data[data["Date"] <= current_date]
    
    if len(data) == 0:
        st.error("No valid data remaining after cleaning. Please check your CSV file.")
        st.stop()

    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(data))
    with col2:
        st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
    with col3:
        if "Close" in data.columns:
            st.metric("Avg Close Price", f"${data['Close'].mean():.2f}")
    with col4:
        if "Close" in data.columns:
            st.metric("Price Range", f"${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    st.dataframe(data.head(10), use_container_width=True)

    # -----------------------------
    # Sidebar Settings
    # -----------------------------
    st.sidebar.header("⚙️ Model Settings")
    
    # Adaptive window size based on data length
    max_window = min(10, len(data) // 3)  # Reduced max window for small dataset
    window_size = st.sidebar.slider("Window Size (Days)", 2, max_window, min(3, max_window))
    
    # Check if we have enough data for the window size
    if len(data) <= window_size:
        st.error(f"Not enough data points ({len(data)}) for window size {window_size}. Please upload more data or reduce window size.")
        st.stop()
    
    # Adaptive epochs based on data size
    epochs = st.sidebar.slider("Epochs", 5, 50, 30)  # Reduced max epochs for small dataset
    batch_size = st.sidebar.slider("Batch Size", 2, 16, 8)  # Reduced batch size for small dataset
    
    # RNN specific settings
    rnn_units = st.sidebar.slider("RNN Units", 16, 64, 32, 8)  # Reduced max units for small dataset
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    
    # Add option for future predictions
    future_days = st.sidebar.number_input("Predict Future Days", min_value=1, max_value=10, value=3)  # Reduced max future days

    # -----------------------------
    # Preprocessing
    # -----------------------------
    close_prices = data["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # -----------------------------
    # Create Sequences
    # -----------------------------
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, window_size)

    if len(X) == 0:
        st.error("Not enough data for selected window size. Please reduce window size.")
        st.stop()

    # Adaptive split ratio - use more for training since dataset is small
    split_ratio = 0.7  # Fixed at 70% training for small dataset
    split = int(split_ratio * len(X))

    if split == 0 or split >= len(X):
        st.error("Dataset too small to split. Please use larger dataset or reduce window size.")
        st.stop()

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Display data split information
    st.info(f"Data Split: {len(X_train)} training samples, {len(X_test)} testing samples")

    # -----------------------------
    # Build RNN Model
    # -----------------------------
    model = Sequential()
    
    # Simpler model for small dataset
    model.add(SimpleRNN(
        units=rnn_units,
        activation="tanh",
        return_sequences=False,  # Single layer for small dataset
        input_shape=(window_size, 1)
    ))
    model.add(Dropout(dropout_rate))
    
    # Dense layers for output
    model.add(Dense(rnn_units // 2, activation="relu"))
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mse", metrics=['mae'])

    st.subheader("🔄 Training Model...")
    
    with st.spinner("Training in progress..."):
        # Train without validation split for very small datasets
        if len(X_train) < 10:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=min(batch_size, len(X_train)),  # Ensure batch size doesn't exceed data
                verbose=0,
                shuffle=False
            )
        else:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0,
                shuffle=False
            )

    st.success("✅ Model Trained Successfully!")

    # -----------------------------
    # Training History Visualization
    # -----------------------------
    if len(history.history['loss']) > 0:
        st.subheader("📉 Training History")
        
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss', mode='lines'))
        if 'val_loss' in history.history:
            fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss', mode='lines'))
        
        fig_history.update_layout(
            title='Model Loss Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x'
        )
        st.plotly_chart(fig_history, use_container_width=True)

    # -----------------------------
    # Predictions
    # -----------------------------
    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    # Match test dates
    test_dates = data["Date"].iloc[window_size + split:].reset_index(drop=True)

    # Ensure lengths match
    min_len = min(len(test_dates), len(actual), len(predictions))
    
    results_df = pd.DataFrame({
        "Date": test_dates[:min_len],
        "Actual": actual.flatten()[:min_len],
        "Predicted": predictions.flatten()[:min_len]
    })

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    if len(results_df) > 0:
        st.subheader("📊 Evaluation Metrics")
        
        try:
            mae = mean_absolute_error(results_df["Actual"], results_df["Predicted"])
            mse = mean_squared_error(results_df["Actual"], results_df["Predicted"])
            rmse = np.sqrt(mse)
            
            # Avoid division by zero in MAPE
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((results_df["Actual"] - results_df["Predicted"]) / results_df["Actual"])) * 100
                if np.isnan(mape) or np.isinf(mape):
                    mape = 0
            
            r2 = r2_score(results_df["Actual"], results_df["Predicted"])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("MAE", f"${mae:.2f}")
            with col2:
                st.metric("MSE", f"{mse:.2f}")
            with col3:
                st.metric("RMSE", f"${rmse:.2f}")
            with col4:
                st.metric("MAPE", f"{mape:.2f}%")
            with col5:
                st.metric("R² Score", f"{r2:.4f}")
        except Exception as e:
            st.warning(f"Could not calculate some metrics: {e}")

        # -----------------------------
        # Plot Graph
        # -----------------------------
        st.subheader("📈 Prediction Graph")
        
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=results_df["Date"],
            y=results_df["Actual"],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=results_df["Date"],
            y=results_df["Predicted"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Stock Prices',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Future Predictions
        # -----------------------------
        if future_days > 0:
            st.subheader("🔮 Future Price Predictions")
            
            # Get the last window_size days of data
            last_sequence = scaled_data[-window_size:].reshape(1, window_size, 1)
            
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            for i in range(future_days):
                next_pred = model.predict(current_sequence, verbose=0)
                future_predictions.append(next_pred[0, 0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Inverse transform predictions
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)
            
            # Create future dates
            last_date = data["Date"].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
            
            future_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Price": future_predictions.flatten()
            })
            
            # Display future predictions
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_future = go.Figure()
                
                # Add historical data
                fig_future.add_trace(go.Scatter(
                    x=data["Date"].iloc[-min(30, len(data)):],
                    y=data["Close"].iloc[-min(30, len(data)):],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add future predictions
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions.flatten(),
                    mode='lines+markers',
                    name='Future Prediction',
                    line=dict(color='green', dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig_future.update_layout(
                    title=f'Next {future_days} Days Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x'
                )
                
                st.plotly_chart(fig_future, use_container_width=True)
            
            with col2:
                st.subheader("📅 Prediction Details")
                st.dataframe(future_df, use_container_width=True)
                
                # Summary statistics
                st.metric("Predicted End Price", f"${future_predictions[-1][0]:.2f}")
                price_change = future_predictions[-1][0] - data["Close"].iloc[-1]
                change_percent = (price_change / data["Close"].iloc[-1]) * 100
                st.metric("Expected Change", 
                         f"${price_change:.2f}",
                         f"{change_percent:.2f}%")

        # -----------------------------
        # Prediction Table
        # -----------------------------
        st.subheader("📋 Detailed Results Table")
        
        display_df = results_df.copy()
        display_df['Actual'] = display_df['Actual'].map('${:.2f}'.format)
        display_df['Predicted'] = display_df['Predicted'].map('${:.2f}'.format)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="stock_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a CSV file to start the analysis.")
    
    # Show sample format
    with st.expander("📋 Expected CSV Format"):
        st.code("""
Date,Close
2024-01-01,100.00
2024-01-02,102.19
2024-01-03,103.11
        """)
