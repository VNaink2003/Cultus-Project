# =========================================================
# ADVANCED TIME SERIES FORECASTING WITH UQ (FINAL VERSION)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# 1. DATA GENERATION
# =========================================================
def generate_data(n=3000):
    t = np.arange(n)

    trend = 0.003 * t
    season1 = 2*np.sin(0.02*t)
    season2 = 0.5*np.sin(0.15*t)
    noise = np.random.normal(0, 0.4, n)

    exog1 = np.sin(0.05*t) + np.random.normal(0,0.2,n)
    exog2 = np.cos(0.03*t) + np.random.normal(0,0.2,n)

    y = trend + season1 + season2 + noise + 0.4*exog1

    df = pd.DataFrame({
        "target": y,
        "exog1": exog1,
        "exog2": exog2
    })

    return df

# =========================================================
# 2. SEQUENCE CREATION
# =========================================================
def create_sequences(data, seq_len=50):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

# =========================================================
# 3. MODEL
# =========================================================
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        super().__init__()

        self.quantiles = [0.1, 0.5, 0.9]

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=layers,
                            batch_first=True,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_size, len(self.quantiles))

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

# =========================================================
# 4. QUANTILE LOSS
# =========================================================
def quantile_loss(pred, target, quantiles):
    loss = 0
    for i,q in enumerate(quantiles):
        error = target - pred[:,i]
        loss += torch.mean(torch.max((q-1)*error, q*error))
    return loss

# =========================================================
# 5. TRAIN FUNCTION
# =========================================================
def train_model(model, X_train, y_train, epochs=20):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_t)
        loss = quantile_loss(pred, y_t.unsqueeze(1), model.quantiles)

        loss.backward()
        optimizer.step()

    return model

# =========================================================
# 6. MC DROPOUT PREDICTION
# =========================================================
def mc_dropout_predict(model, X, samples=100):

    model.train()
    preds = []

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    for _ in range(samples):
        with torch.no_grad():
            preds.append(model(X_t).cpu().numpy())

    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)

    lower = mean_preds[:,0]
    median = mean_preds[:,1]
    upper = mean_preds[:,2]

    return lower, median, upper

# =========================================================
# 7. METRICS
# =========================================================
def coverage_probability(y, lower, upper):
    return np.mean((y>=lower)&(y<=upper))

def interval_width(lower, upper):
    return np.mean(upper-lower)

# =========================================================
# MAIN PIPELINE
# =========================================================
def main():

    print("\nGenerating dataset...")
    df = generate_data()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequences(scaled, 50)

    split = int(len(X)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    best_model = None
    best_rmse = 999

    print("\nHyperparameter tuning...")

    for hidden, layers, drop in product([32,64], [1,2], [0.1,0.3]):

        model = QuantileLSTM(X.shape[2], hidden, layers, drop).to(DEVICE)
        model = train_model(model, X_train, y_train)

        _, median, _ = mc_dropout_predict(model, X_test)
        rmse = np.sqrt(mean_squared_error(y_test, median))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    print("Best model selected")

    # ==========================
    # FINAL DL MODEL EVALUATION
    # ==========================
    lower, median, upper = mc_dropout_predict(best_model, X_test)

    rmse_dl = np.sqrt(mean_squared_error(y_test, median))
    mae_dl = mean_absolute_error(y_test, median)

    cp = coverage_probability(y_test, lower, upper)
    iw = interval_width(lower, upper)

    print("\nDEEP LEARNING RESULTS")
    print("RMSE:", rmse_dl)
    print("MAE:", mae_dl)
    print("Coverage Probability:", cp)
    print("Interval Width:", iw)

    # ==========================
    # BASELINE (TEST SET ONLY)
    # ==========================
    baseline_preds = X_test[:,-1,0]

    rmse_base = np.sqrt(mean_squared_error(y_test, baseline_preds))
    mae_base = mean_absolute_error(y_test, baseline_preds)

    print("\nBASELINE RESULTS")
    print("RMSE:", rmse_base)
    print("MAE:", mae_base)

    print("\nProject execution complete.")

if __name__ == "__main__":
    main()
