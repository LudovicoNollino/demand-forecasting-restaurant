import matplotlib.pyplot as plt

def plot_forecasting(
    col_name,
    y_train_orig, y_val_orig, y_test_orig,
    val_pred_orig, test_pred_orig, future_pred_orig,
    future_idx_orig=None,
    alpha_zone=0.12
):
    plt.figure(figsize=(16, 9))

    # Background zone
    if len(y_val_orig) > 0:
        plt.axvspan(y_val_orig.index[0], y_val_orig.index[-1], color='orange', alpha=alpha_zone, label='Validation Period')
    if len(y_test_orig) > 0:
        plt.axvspan(y_test_orig.index[0], y_test_orig.index[-1], color='green', alpha=alpha_zone, label='Test Period')
    if future_pred_orig is not None and len(future_pred_orig) > 0:
        idx_fut = future_idx_orig if future_idx_orig is not None else future_pred_orig.index
        plt.axvspan(idx_fut[0], idx_fut[-1], color='red', alpha=alpha_zone, label='Future Forecast Period')

    plt.plot(y_train_orig.index, y_train_orig, color='blue', label='Train Obs', linewidth=2)
    plt.plot(y_val_orig.index, y_val_orig, color='orange', label='Val Obs', linewidth=2)
    plt.plot(y_test_orig.index, y_test_orig, color='green', label='Test Obs', linewidth=2)

    # Previsions
    if val_pred_orig is not None and len(val_pred_orig) > 0:
        plt.plot(val_pred_orig.index, val_pred_orig, '--', color='red', label='Val Forecast', linewidth=2.5)
    if test_pred_orig is not None and len(test_pred_orig) > 0:
        plt.plot(test_pred_orig.index, test_pred_orig, '--', color='purple', label='Test Forecast', linewidth=2.5)
    if future_pred_orig is not None and len(future_pred_orig) > 0:
        idx_fut = future_idx_orig if future_idx_orig is not None else future_pred_orig.index
        plt.plot(idx_fut, future_pred_orig, '-.', color='black', label='Future Forecast', linewidth=2.5)

    # Vertical split lines
    plt.axvline(y_train_orig.index[-1], color="black", linestyle=":", label="Split Train/Val")
    plt.axvline(y_val_orig.index[-1], color="black", linestyle=":", label="Split Val/Test")
    plt.axvline(y_test_orig.index[-1], color="black", linestyle=":", label="Split Test/Future")

    plt.title(f"{col_name} - Original Scale")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
