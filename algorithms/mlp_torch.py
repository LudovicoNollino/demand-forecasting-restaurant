import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from dataset_manipulation.preprocessing import inverse_transform_predictions_forecast

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, activation='relu'):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.layers(x)

def create_sliding_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def invert_preds(preds, preprocess_params):
    if (
        preprocess_params.get("boxcox_applied", False)
        or preprocess_params.get("kalman_applied", False)
        or (preprocess_params.get("scaler", None) is not None)
    ):
        return inverse_transform_predictions_forecast(pd.Series(preds), preprocess_params).values
    return preds

def train_mlp_model(X_train, y_train, X_val, y_val, hidden_dim1, hidden_dim2, lr=0.001,
                    activation='relu', n_epochs=200, batch_size=1, print_every=20, verbose=True):
    model = SimpleMLP(X_train.shape[1], hidden_dim1, hidden_dim2, activation=activation)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train).unsqueeze(1)
    X_vl = torch.FloatTensor(X_val)
    y_vl = torch.FloatTensor(y_val).unsqueeze(1)
    best_weights = None
    best_val_loss = np.inf

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(len(X_tr))
        train_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            Xb = X_tr[idx]
            yb = y_tr[idx]
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(X_tr)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl)
            val_loss = loss_fn(val_pred, y_vl).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict()
        if verbose and (epoch % print_every == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}")

    # Ripristina i pesi migliori e metti in eval
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model

def mlp_grid_search(
    X_train, y_train, X_val, y_val,
    hidden1_grid, hidden2_grid, lr_grid, activations=['relu'],
    n_epochs_grid=[200], batch_size=1, print_every=50
):
    results = []
    best_val_loss = np.inf
    best_params = None
    best_model = None
    count = 0
    total = len(hidden1_grid) * len(hidden2_grid) * len(lr_grid) * len(activations) * len(n_epochs_grid)
    for h1 in hidden1_grid:
        for h2 in hidden2_grid:
            for lr in lr_grid:
                for act in activations:
                    for n_epochs in n_epochs_grid:
                        print(f"Grid Search {count+1}/{total}: h1={h1}, h2={h2}, lr={lr}, act={act}, epochs={n_epochs}")
                        model = train_mlp_model(
                            X_train, y_train, X_val, y_val,
                            hidden_dim1=h1,
                            hidden_dim2=h2,
                            lr=lr,
                            activation=act,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            print_every=print_every,
                            verbose=False  # Nessun print durante gridsearch, sennò è illeggibile!
                        )
                        # Validation loss finale
                        model.eval()
                        X_vl = torch.FloatTensor(X_val)
                        y_vl = torch.FloatTensor(y_val).unsqueeze(1)
                        with torch.no_grad():
                            val_pred = model(X_vl)
                            val_loss = nn.MSELoss()(val_pred, y_vl).item()
                        results.append({
                            'hidden1': h1, 'hidden2': h2, 'lr': lr,
                            'activation': act, 'epochs': n_epochs, 'val_loss': val_loss
                        })
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = (h1, h2, lr, act, n_epochs, batch_size)
                            best_model = model
                        count += 1
    print("\nGrid Search Results (Top 5 by val_loss):")
    grid_df = pd.DataFrame(results).sort_values("val_loss").head(5)
    print(grid_df)
    return best_model, best_params, results

def fit_mlp_model(
    data_dict,
    window_size=7,
    hidden_dim1=16,
    hidden_dim2=8,
    lr=0.001,
    activation='relu',
    n_epochs=200,
    batch_size=1,
    print_every=20,
    future_steps=30,
    grid_search=False,
    grid_params=None,
    col_name="Serie",
    verbose=True
):
    # Preprocessing & Windowing
    series_proc = pd.concat([data_dict["train"], data_dict["val"], data_dict["test"]]).values.astype(np.float32)
    n_train, n_val, n_test = len(data_dict["train"]), len(data_dict["val"]), len(data_dict["test"])
    preprocess_params = data_dict['preprocess_params']

    X_train, y_train = create_sliding_windows(series_proc[:n_train+window_size], window_size)
    X_val, y_val = create_sliding_windows(series_proc[n_train:n_train+n_val+window_size], window_size)
    X_test, y_test = create_sliding_windows(series_proc[n_train+n_val:], window_size)
    if verbose:
        print(f"Train samples shape: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Window: {window_size}")

    # Grid search o modello singolo
    if grid_search and grid_params is not None:
        model, best_params, grid = mlp_grid_search(
            X_train, y_train, X_val, y_val, **grid_params
        )
    else:
        model = train_mlp_model(
            X_train, y_train, X_val, y_val,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            lr=lr,
            activation=activation,
            n_epochs=n_epochs,
            batch_size=batch_size,
            print_every=print_every,
            verbose=verbose
        )
        best_params = (hidden_dim1, hidden_dim2, lr, activation, n_epochs, batch_size)
        grid = None

    # Predict e inversione scala
    model.eval()
    X_tr = torch.FloatTensor(X_train)
    X_vl = torch.FloatTensor(X_val)
    X_ts = torch.FloatTensor(X_test)
    with torch.no_grad():
        y_tr_pred = model(X_tr).numpy().flatten()
        y_vl_pred = model(X_vl).numpy().flatten()
        y_ts_pred = model(X_ts).numpy().flatten()
    y_train_inv = invert_preds(y_tr_pred, preprocess_params)
    y_val_inv = invert_preds(y_vl_pred, preprocess_params)
    y_test_inv = invert_preds(y_ts_pred, preprocess_params)

    # RMSE e AIC
    train_rmse = mean_squared_error(y_train, y_tr_pred) ** 0.5
    val_rmse = mean_squared_error(y_val, y_vl_pred) ** 0.5
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rss = np.sum((y_train - y_tr_pred) ** 2)
    aic = len(y_train) * np.log(rss / len(y_train)) + 2 * k

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val   RMSE: {val_rmse:.4f}")
    print(f"AIC: {aic:.2f}")

    # Forecast autoregressivo futuro
    input_seq = series_proc[-window_size:].copy()
    future_forecast = []
    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            inp = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            pred = model(inp).item()
            future_forecast.append(pred)
            input_seq = np.roll(input_seq, -1)
            input_seq[-1] = pred
    print("Forecast autoregressivo futuro (30 passi):")
    print(np.array(future_forecast))
    future_forecast_inv = invert_preds(np.array(future_forecast), preprocess_params)

    # Plot risultati
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    idx = pd.concat([data_dict["train_orig"], data_dict["val_orig"], data_dict["test_orig"]]).index
    plt.plot(data_dict["train_orig"].index, data_dict["train_orig"], label="Train Obs", color="blue")
    plt.plot(data_dict["val_orig"].index, data_dict["val_orig"], label="Val Obs", color="orange")
    plt.plot(data_dict["test_orig"].index, data_dict["test_orig"], label="Test Obs", color="green")
    plt.plot(idx[window_size:window_size+len(y_train_inv)], y_train_inv, '--', color="blue", label="Train Predict")
    plt.plot(idx[len(data_dict["train"])+window_size:len(data_dict["train"])+window_size+len(y_val_inv)], y_val_inv, '--', color="orange", label="Val Predict")
    plt.plot(idx[len(data_dict["train"])+len(data_dict["val"])+window_size:len(data_dict["train"])+len(data_dict["val"])+window_size+len(y_test_inv)], y_test_inv, '--', color="green", label="Test Predict")
    # Future
    last_idx = data_dict["test_orig"].index[-1]
    if isinstance(last_idx, pd.Timestamp):
        future_idx = pd.date_range(start=last_idx + pd.Timedelta(days=1), periods=future_steps)
    else:
        future_idx = np.arange(last_idx + 1, last_idx + 1 + future_steps)
    plt.plot(future_idx, future_forecast_inv, 'r--', label="Future Forecast (orig)")
    plt.title(f"{col_name} - Original Scale (val/test e 30gg futuri) [MLP]")
    plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    idx_proc = np.arange(len(series_proc))
    plt.plot(idx_proc[:len(data_dict["train"])], series_proc[:len(data_dict["train"])], label="Train Obs (proc)", color="blue")
    plt.plot(idx_proc[len(data_dict["train"]):len(data_dict["train"])+len(data_dict["val"])], series_proc[len(data_dict["train"]):len(data_dict["train"])+len(data_dict["val"])], label="Val Obs (proc)", color="orange")
    plt.plot(idx_proc[len(data_dict["train"])+len(data_dict["val"]):], series_proc[len(data_dict["train"])+len(data_dict["val"]):], label="Test Obs (proc)", color="green")
    plt.plot(idx_proc[window_size:window_size+len(y_tr_pred)], y_tr_pred, '--', color="blue", label="Train Predict (proc)")
    plt.plot(idx_proc[len(data_dict["train"])+window_size:len(data_dict["train"])+window_size+len(y_vl_pred)], y_vl_pred, '--', color="orange", label="Val Predict (proc)")
    plt.plot(idx_proc[len(data_dict["train"])+len(data_dict["val"])+window_size:len(data_dict["train"])+len(data_dict["val"])+window_size+len(y_ts_pred)], y_ts_pred, '--', color="green", label="Test Predict (proc)")
    plt.plot(np.arange(len(series_proc), len(series_proc)+future_steps), future_forecast, 'r--', label="Future Forecast (proc)")
    plt.title(f"{col_name} - Processed Scale (val/test e 30gg futuri) [MLP]")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, best_params, grid, y_val_inv, y_test_inv, future_forecast_inv
