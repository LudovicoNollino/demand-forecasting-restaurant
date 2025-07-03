import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calc_metrics(y_true, y_pred):
    """
    Calculates MAPE, MAE, RMSE always aligning series with the same minimum length at the end.
    Returns a dictionary {metric_name: value}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    minlen = min(len(y_true), len(y_pred))
    if minlen == 0:
        return {"MAPE": np.nan, "MAD": np.nan, "RMSE": np.nan}
    y_true = y_true[-minlen:]
    y_pred = y_pred[-minlen:]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return { "MAPE": mape, "MAE": mae, "RMSE": rmse }

def get_metrics_table(val_true, val_pred, test_true, test_pred, round_n=4):
    """
    Returns an HTML table with evaluation metrics for validation and test sets.
    """
    val_metrics = calc_metrics(val_true, val_pred)
    test_metrics = calc_metrics(test_true, test_pred)
    html = "<br><b>Model evaluation metrics (validation / test):</b><br>"
    html += "<table border='1' cellpadding='3'><tr><th>Set</th>"
    for m in val_metrics.keys():
        html += f"<th>{m}</th>"
    html += "</tr>"
    html += "<tr><td>Validation</td>" + "".join([f"<td>{val_metrics[k]:.{round_n}f}</td>" for k in val_metrics.keys()]) + "</tr>"
    html += "<tr><td>Test</td>" + "".join([f"<td>{test_metrics[k]:.{round_n}f}</td>" for k in test_metrics.keys()]) + "</tr>"
    html += "</table>"
    return html
