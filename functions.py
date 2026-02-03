import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as sklearn_roc_curve, auc, confusion_matrix
from scipy.stats import mannwhitneyu


def calculate_roc_metrics(cases, controls, name="Feature"):
    y = pd.concat([pd.Series(1, index=cases.index),
                   pd.Series(0, index=controls.index)])
    scores = pd.concat([cases, controls])

    fpr, tpr, thresholds = sklearn_roc_curve(y, scores)
    auc_val = auc(fpr, tpr)

    J_scores = tpr - fpr
    best_idx = np.argmax(J_scores)
    best_j = J_scores[best_idx]
    sens = tpr[best_idx]
    spec = 1 - fpr[best_idx]
    thresh = thresholds[best_idx]

    y_pred = (scores >= thresh).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / total
    lr_plus = sens / fpr[best_idx] if fpr[best_idx] > 0 else 0
    lr_minus = (1 - sens) / spec if spec > 0 else 0
    odds_ratio = lr_plus / lr_minus if lr_minus > 0 else 0

    expected_correct = ((tp+fn)*(tp+fp) + (tn+fp)*(tn+fn)) / total**2
    kappa = (accuracy - expected_correct) / (1 - expected_correct) if expected_correct < 1 else 0

    statistic, p_value = mannwhitneyu(cases, controls, alternative='two-sided')

    st.write(f"## {name}")
    
    metrics_data = {
        'Parameters': [
            'Threshold',
            'Sensitivity',
            'Specificity',
            'AUC',
            'PPV',
            'NPV',
            'Accuracy',
            'LR+',
            'LR-',
            'OR',
            'AUC p-value'
        ],
        'Value': [
            f"{thresh:.4f}",
            f"{sens:.4f}",
            f"{spec:.4f}",
            f"{auc_val:.4f}",
            f"{ppv:.4f}",
            f"{npv:.4f}",
            f"{accuracy:.4f}",
            f"{lr_plus:.4f}",
            f"{lr_minus:.4f}",
            f"{odds_ratio:.4f}",
            f"{p_value:.4f}"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    csv = metrics_df.to_csv(index=False)
    st.download_button("Download Metrics", csv, f"{name}_metrics.csv", "text/csv")
    
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(fpr, tpr, linewidth=2.5, label=f'AUC={auc_val:.3f}')
    ax.scatter(fpr[best_idx], tpr[best_idx], c='red', s=120, zorder=5, label=f'J={best_j:.3f}')
    ax.plot([0,1],[0,1],'k--', alpha=0.7)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def create_boxplot(cases, controls, cases_col, controls_col, name="Feature"):
    st.write(f"## {name}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = [cases.dropna(), controls.dropna()]
    bp = ax.boxplot(data_to_plot, labels=[cases_col, controls_col], patch_artist=True)
    
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.set_ylabel('Values')
    ax.set_title(f'{name}')
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close(fig)
    
    summary_data = {
        'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        cases_col: [
            len(cases),
            f"{cases.mean():.4f}",
            f"{cases.median():.4f}",
            f"{cases.std():.4f}",
            f"{cases.min():.4f}",
            f"{cases.max():.4f}"
        ],
        controls_col: [
            len(controls),
            f"{controls.mean():.4f}",
            f"{controls.median():.4f}",
            f"{controls.std():.4f}",
            f"{controls.min():.4f}",
            f"{controls.max():.4f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    csv = summary_df.to_csv(index=False)
    st.download_button("Download Statistics", csv, f"{name}_stats.csv", "text/csv")
    
    statistic, p_value = mannwhitneyu(cases, controls, alternative='two-sided')
    st.write(f"**Mann-Whitney U p-value:** {p_value:.4f}")