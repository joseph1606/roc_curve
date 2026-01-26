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

    print(f"\n{name}")
    print(f"J: {best_j:.4f}")
    print(f"Threshold: {thresh:.4f}")
    print(f"AUC: {auc_val:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"TN: {int(tn)} | FP: {int(fp)} | FN: {int(fn)} | TP: {int(tp)}")
    print(f"PPV: {ppv:.4f} | NPV: {npv:.4f} | Accuracy: {accuracy:.4f}")
    print(f"LR+: {lr_plus:.4f} | LR-: {lr_minus:.4f} | OR: {odds_ratio:.4f}")
    print(f"Kappa: {kappa:.4f} | p-value: {p_value:.4f}\n")

    st.write(name)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("J", f"{best_j:.4f}")
    col2.metric("Threshold", f"{thresh:.4f}")
    col3.metric("AUC", f"{auc_val:.4f}")
    
    col1, col2 = st.columns(2)
    col1.metric("Sensitivity", f"{sens:.4f}")
    col2.metric("Specificity", f"{spec:.4f}")
    
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
    
    col1, col2, col3 = st.columns(3)
    col1.metric("PPV", f"{ppv:.4f}")
    col2.metric("NPV", f"{npv:.4f}")
    col3.metric("Accuracy", f"{accuracy:.4f}")
    
    st.metric("p-value", f"{p_value:.4f}")


sheet = st.text_input("Sheet name", key="Sheet Name" )
cases_col = st.text_input("Cases column", key="case")
controls_col = st.text_input("Controls column", key="control")
graph_name = st.text_input("Graph name", key="graph")

uploaded_file = st.file_uploader("Excel file", type='xlsx')

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        st.dataframe(df)
        
        if not cases_col or not controls_col:
            st.warning("Enter both column names")
        elif cases_col not in df.columns:
            st.error(f"Column {cases_col} not found")
        elif controls_col not in df.columns:
            st.error(f"Column {controls_col} not found")
        else:
            cases = df[cases_col].dropna()
            controls = df[controls_col].dropna()
            
            if len(cases) == 0:
                st.error(f"No data in {cases_col}")
            elif len(controls) == 0:
                st.error(f"No data in {controls_col}")
            elif len(cases) < 2 or len(controls) < 2:
                st.error(f"Need at least 2 values in each column")
            else:
                calculate_roc_metrics(cases, controls, graph_name)
                
    except Exception as e:
        st.error(f"Error: {e}")