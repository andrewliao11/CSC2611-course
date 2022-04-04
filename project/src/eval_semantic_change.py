import sys
import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipdb


def main():
    log_path = sys.argv[1]
    outputs = json.load(open(log_path))
    
    influences = defaultdict(lambda: [])
    binary = {}
    grades = {}
    for i in outputs:
        target_word, corpus, influence, binary_gt, grade_gt, n_removed = i
        binary[target_word] = int(binary_gt)
        grades[target_word] = float(grade_gt)

        if influence < 5:
            influences[target_word].append(influence)
            #influences[target_word].append(influence / n_removed)

    
    for k in influences.keys():
        influences[k] = np.mean(influences[k])
        
    
    min_inf = min(influences.values())
    max_inf = max(influences.values())
    best_thres, best_acc = None, -np.inf
    for thres in np.linspace(min_inf, max_inf, 10):
        print("==================================================")
        print(f"Threshold: {thres}")
        keys = np.array(list(influences.keys()))
        pred_inf = np.array([influences[k] for k in keys])
        binary_gt = np.array([binary[k] for k in keys]) == 1
        
        correct = (pred_inf > thres) == binary_gt
        acc = np.mean(correct)
        if acc > best_acc:
            best_thres = thres

        print(f"Accuracy: {acc:.2f}")
        
        print(f"Correct target words   ({correct.sum():2}): {', '.join(keys[correct])}")
        print(f"Incorrect target words ({(~correct).sum():2}): {', '.join(keys[~correct])}")
        

    inf = np.array([influences[k] for k in keys])
    grad = np.array([grades[k] for k in keys])
    corr = spearmanr(inf, grad)[0]
    print(f"Spearman's correlation: {corr}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inf, y=grad, showlegend=False, mode="markers"))
    fig.update_layout(template="plotly_white", width=400, height=400, margin=dict(t=10, l=50, b=50, r=10))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_xaxes(title="Pred Influences")
    fig.update_yaxes(title="GT values")

    fig.write_image("influences_grade_scatter.png")
    fig.write_image("influences_grade_scatter.pdf")

    pred_inf = np.array([influences[k] for k in keys])
    binary_gt = np.array([binary[k] for k in keys]) == 1
    
    pred = pred_inf > thres
    correct = pred == binary_gt


    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("w/ Semantic change", "w/o Semantic change"))

    x = keys[binary_gt]
    y = pred_inf[binary_gt]
    idx = y.argsort()[::-1]
    fig.add_trace(go.Bar(x=x[idx], y=y[idx], showlegend=False), row=1, col=1)


    x = keys[~binary_gt]
    y = pred_inf[~binary_gt]
    idx = y.argsort()[::-1]
    fig.add_trace(go.Bar(x=x[idx], y=y[idx], showlegend=False), row=1, col=2)

    fig.update_layout(template="plotly_white", width=700, height=400, margin=dict(t=30, l=50, b=50, r=10))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_yaxes(title="Pred Influences")

    fig.write_image("influences_binary.png")
    fig.write_image("influences_binary.pdf")

    
if __name__ == "__main__":
    main()
