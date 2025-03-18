import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Function to load and preprocess the data
def load_data(file_path):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(file_path)
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Convert date-related columns to proper format
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
    
    # Ensure numeric columns are properly typed
    volume_cols = ['YTD Value', 'Average Volume for Age Groups', 'Sum of Volume']
    for col in volume_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Function to create sequences for time series forecasting
def create_sequences(data, seq_length):
    """Create input sequences and targets for time series forecasting."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to build and train the LSTM model
def build_lstm_model(seq_length, n_features=1):
    """Build and compile an LSTM neural network model."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Function to train model and generate forecasts
def train_and_forecast(data, group, seq_length=7, forecast_periods=30):
    """Train the model and generate forecasts for a specific age group."""
    # Filter data for the selected group
    group_data = data[data['cE_Description'] == group].sort_values('Date')
    
    # Extract the target variable
    volume_data = group_data['Average Volume for Age Groups'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(volume_data)
    
    # Create sequences
    if len(scaled_data) <= seq_length:
        return None, None, None, f"Not enough data points for group {group}. Need more than {seq_length} points."
    
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data into train and test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train the model
    model = build_lstm_model(seq_length)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
        shuffle=False
    )
    
    # Generate predictions for existing data
    predictions = []
    for i in range(len(X)):
        pred = model.predict(X[i].reshape(1, seq_length, 1), verbose=0)
        predictions.append(pred[0, 0])
    
    # Generate future forecasts
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    future_preds = []
    
    current_sequence = last_sequence.copy()
    for _ in range(forecast_periods):
        next_pred = model.predict(current_sequence, verbose=0)[0, 0]
        future_preds.append(next_pred)
        # Update the sequence for the next prediction
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                     [[next_pred]], 
                                     axis=1)
    
    # Inverse transform to get actual values
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    
    # Create dates for future predictions
    last_date = group_data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_preds))]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_preds
    })
    
    return predictions, future_preds, forecast_df, None

# Function to create interactive dashboard
def create_dashboard(data_path):
    """Create a Dash app for interactive forecasting visualization."""
    # Load data
    df = load_data(data_path)
    
    # Get unique age groups
    age_groups = sorted(df['cE_Description'].unique())
    
    # Initialize the Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define the layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Age Group Volume Forecasting Dashboard", className="text-center my-4"),
                html.Hr(),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Age Group:"),
                dcc.Dropdown(
                    id='age-group-dropdown',
                    options=[{'label': group, 'value': group} for group in age_groups],
                    value=age_groups[0] if age_groups else None
                ),
            ], width={"size": 6, "offset": 3}, className="mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading",
                    type="circle",
                    children=[
                        dcc.Graph(id='forecast-graph', style={'height': '600px'})
                    ]
                )
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='forecast-summary', className="mt-4")
            ], width=12)
        ])
    ], fluid=True)
    
    # Define callback to update the graph based on selected age group
    @app.callback(
        [Output('forecast-graph', 'figure'),
         Output('forecast-summary', 'children')],
        [Input('age-group-dropdown', 'value')]
    )
    def update_graph(selected_group):
        if not selected_group:
            return go.Figure(), html.Div("Please select an age group")
        
        # Filter data for the selected group
        group_data = df[df['cE_Description'] == selected_group].sort_values('Date')
        
        if len(group_data) < 10:  # Minimum data points needed
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data points for forecasting",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig, html.Div("Not enough data for forecasting. Need at least 10 data points.")
        
        # Train model and generate forecasts
        predictions, future_preds, forecast_df, error_msg = train_and_forecast(df, selected_group)
        
        if error_msg:
            fig = go.Figure()
            fig.add_annotation(
                text=error_msg,
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig, html.Div(error_msg)
        
        # Create the figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=group_data['Date'],
                y=group_data['Average Volume for Age Groups'],
                mode='lines+markers',
                name='Actual Volume',
                line=dict(color='blue')
            )
        )
        
        # Add historical predictions
        offset = len(group_data) - len(predictions)
        fig.add_trace(
            go.Scatter(
                x=group_data['Date'][offset:],
                y=predictions,
                mode='lines',
                name='Model Fit',
                line=dict(color='green', dash='dot')
            )
        )
        
        # Add future predictions
        fig.add_trace(
            go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Volume Forecast for {selected_group}",
            xaxis_title="Date",
            yaxis_title="Average Volume",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Create forecast summary
        latest_actual = group_data['Average Volume for Age Groups'].iloc[-1]
        next_forecast = forecast_df['Forecast'].iloc[0]
        forecast_trend = ((forecast_df['Forecast'].iloc[-1] / latest_actual) - 1) * 100
        
        summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Latest Actual Volume"),
                    dbc.CardBody(html.H4(f"{latest_actual:.2f}"))
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Next Period Forecast"),
                    dbc.CardBody(html.H4(f"{next_forecast:.2f}"))
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"{len(forecast_df)}-Day Trend"),
                    dbc.CardBody(html.H4(f"{forecast_trend:.2f}%"))
                ])
            ], width=4)
        ])
        
        return fig, summary
    
    return app

# Main execution function
def main(data_path):
    """Main function to run the forecasting app."""
    app = create_dashboard(data_path)
    return app

if __name__ == "__main__":
    # Replace with your CSV file path
    data_file = "data.csv"
    app = main(data_file)
    app.run_server(debug=True)
