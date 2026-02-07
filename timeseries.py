# ==============================================
# ADVANCED TIME SERIES FORECASTING PROJECT
# LSTM + UNCERTAINTY QUANTIFICATION
# ==============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

# ==============================================
# 1. DATA GENERATION (complex non-stationary)
# ==============================================
def generate_time_series(n_steps=2500):
    t = np.arange(n_steps)

    trend = 0.005 * t
    season1 = 2 * np.sin(0.02 * t)
    season2 = 0.5 * np.sin(0.15 * t)
    noise = np.random.normal(0, 0.5, n_steps)

    exog1 = np.sin(0.03 * t) + np.random.normal(0, 0.2, n_steps)
    exog2 = np.cos(0.05 * t) + np.random.normal(0, 0.2, n_steps)

    y = trend + season1 + season2 + noise + 0.3 * exog1

    df = pd.DataFrame({
        "target": y,
        "exog1": exog1,
        "exog2": exog2
    })

    return df

# ==============================================
# 2. DATA PREP
# ==============================================
def create_sequences(data, seq_len=40, horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+horizon, 0])
    return np.array(X), np.array(y)

# ==============================================
# 3. MODEL
# ==============================================
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ==============================================
# 4. LOSS (Pinball)
# ==============================================
def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

# ==============================================
# 5. TRAIN FUNCTION
# ==============================================
def train_model(model, X_train, y_train, epochs=20, lr=0.001):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_t)
        loss = quantile_loss(preds, y_t, model.quantiles)
        loss.backward()
        optimizer.step()

    return model

# ==============================================
# 6. MC DROPOUT PREDICTION
# ==============================================
def mc_dropout_predict(model, X, n_samples=50):
    model.train()
    preds = []
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    for _ in range(n_samples):
        with torch.no_grad():
            preds.append(model(X_t).cpu().numpy())

    preds = np.stack(preds)
    return preds.mean(axis=0)

# ==============================================
# 7. METRICS
# ==============================================
def coverage_probability(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))

def interval_width(lower, upper):
    return np.mean(upper - lower)

# ==============================================
# 8. BASELINE MODEL
# ==============================================
def naive_forecast(series):
    return series[:-1]

# ==============================================
# MAIN PIPELINE
# ==============================================
def main():

    df = generate_time_series()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    seq_len = 40
    X, y = create_sequences(scaled, seq_len)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    quantiles = [0.1, 0.5, 0.9]

    best_model = None
    best_loss = 9999

    # Hyperparameter grid
    hidden_sizes = [32, 64]
    layers = [1, 2]
    dropouts = [0.1, 0.3]

    for hs, nl, dr in product(hidden_sizes, layers, dropouts):

        model = QuantileLSTM(
            input_size=X.shape[2],
            hidden_size=hs,
            num_layers=nl,
            dropout=dr,
            quantiles=quantiles
        )

        model = train_model(model, X_train, y_train, epochs=15)

        preds = mc_dropout_predict(model, X_test)
        loss = mean_squared_error(y_test[:,0], preds[:,1])

        if loss < best_loss:
            best_loss = loss
            best_model = model

    print("Best model selected")

    # FINAL PREDICTION
    preds = mc_dropout_predict(best_model, X_test)

    lower = preds[:,0]
    median = preds[:,1]
    upper = preds[:,2]

    y_true = y_test[:,0]

    rmse = np.sqrt(mean_squared_error(y_true, median))
    mae = mean_absolute_error(y_true, median)

    cp = coverage_probability(y_true, lower, upper)
    iw = interval_width(lower, upper)

    print("\n===== FINAL METRICS =====")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Coverage Probability:", cp)
    print("Interval Width:", iw)

    # Baseline
    baseline = naive_forecast(df["target"].values)
    baseline_rmse = np.sqrt(mean_squared_error(df["target"].values[1:], baseline))
    print("Baseline RMSE:", baseline_rmse)

if __name__ == "__main__":
    main()
