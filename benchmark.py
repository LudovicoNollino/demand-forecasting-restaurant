import numpy as np
import datetime
from utils import calc_metrics

def create_global_report(all_reports):
    md_content = "# DEMAND FORECASTING RESULTS\n"
    md_content += f"**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for rep in all_reports:
        target = rep['target']
        data_dict = rep['data_dict']
        val_pred = rep['val_pred']
        test_pred = rep['test_pred']
        val_pred_mlp = rep['val_pred_mlp']
        test_pred_mlp = rep['test_pred_mlp']
        val_pred_xgb = rep['val_pred_xgb']
        test_pred_xgb = rep['test_pred_xgb']

        val_orig = data_dict['val_orig'][val_pred.index]
        test_orig = data_dict['test_orig'][test_pred.index]

        models_metrics = {}

        # SARIMAX
        if val_pred is not None and test_pred is not None:
            val_metrics = calc_metrics(val_orig.values, val_pred.values)
            test_metrics = calc_metrics(test_orig.values, test_pred.values)
            models_metrics['SARIMAX'] = {'val': val_metrics, 'test': test_metrics}

        # MLP
        if val_pred_mlp is not None and test_pred_mlp is not None:
            val_metrics = calc_metrics(val_orig.values, val_pred_mlp.values)
            test_metrics = calc_metrics(test_orig.values, test_pred_mlp.values)
            models_metrics['MLP'] = {'val': val_metrics, 'test': test_metrics}

        # XGBoost
        if val_pred_xgb is not None and test_pred_xgb is not None:
            val_metrics = calc_metrics(val_orig.values, val_pred_xgb.values)
            test_metrics = calc_metrics(test_orig.values, test_pred_xgb.values)
            models_metrics['XGBoost'] = {'val': val_metrics, 'test': test_metrics}

        # Section for each target/column
        md_content += f"\n\n---\n\n## Target: **{target}**\n"
        md_content += f"- **Train:** {len(data_dict['train'])}\n"
        md_content += f"- **Validation:** {len(data_dict['val'])}\n"
        md_content += f"- **Test:** {len(data_dict['test'])}\n"
        md_content += f"- **Total:** {len(data_dict['train']) + len(data_dict['val']) + len(data_dict['test'])}\n\n"

        # Metrics table per model
        md_content += "| Set | Model | RMSE | MAE | MAPE (%) |\n|-----|-------|------|-----|----------|\n"
        for split in ["val", "test"]:
            for model in ['SARIMAX', 'MLP', 'XGBoost']:
                if model in models_metrics:
                    md = models_metrics[model][split]
                    md_content += f"| {'Validation' if split == 'val' else 'Test'} | {model} | {md['RMSE']:.4f} | {md['MAE']:.4f} | {md['MAPE']:.2f} |\n"

        # Best model on test set
        md_content += "\n**Best model (test RMSE):** "
        test_sorted = sorted(models_metrics.items(), key=lambda x: x[1]['test']['RMSE'])
        best_model = test_sorted[0]
        md_content += f"{best_model[0]} with RMSE={best_model[1]['test']['RMSE']:.4f}\n"

    # Save in the project folder
    import os
    out_path = os.path.join(os.path.dirname(__file__), 'model_results_summary_TOTAL.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Global report generated: {out_path}")
