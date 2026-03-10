"""BotTrainer – Plotly chart helpers"""
import numpy as np
import plotly.graph_objects as go
import pandas as pd

BG   = "#0E1117"
BG2  = "#161B27"
GRID = "#1E2A3A"
FG   = "#E8ECF4"
PALETTE = ["#00C9A7","#7C6AF7","#F5A623","#E84393","#00B4D8","#FF6B6B","#26C485","#FFD166","#118AB2","#F4A261","#A8DADC","#EF476F"]

_L = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG, family="monospace", size=12),
          margin=dict(l=30,r=30,t=45,b=30))

def _ax(): return dict(gridcolor=GRID, zerolinecolor=GRID)

def intent_dist(counts: dict):
    labels, vals = list(counts.keys()), list(counts.values())
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=PALETTE[:len(labels)],
                           text=vals, textposition="outside"))
    fig.update_layout(title="Intent Distribution", xaxis_title="Intent",
                      yaxis_title="Samples", xaxis_tickangle=-30, **_L)
    fig.update_xaxes(**_ax()); fig.update_yaxes(**_ax())
    return fig

def confusion_heatmap(cm: np.ndarray, labels: list):
    norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    text = [[f"{cm[i][j]}<br>({norm[i][j]*100:.0f}%)" for j in range(len(labels))]
            for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, text=text,
                               texttemplate="%{text}", colorscale="Blues",
                               colorbar=dict(tickfont=dict(color=FG))))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted",
                      yaxis_title="Actual", xaxis_tickangle=-30, **_L)
    return fig

def per_intent_bars(report: dict, classes: list):
    valid = [c for c in classes if c in report]
    fig = go.Figure()
    for metric, col in [("precision","#00C9A7"),("recall","#7C6AF7"),("f1-score","#F5A623")]:
        vals = [report[c][metric] for c in valid]
        fig.add_trace(go.Bar(name=metric.title(), x=valid, y=vals,
                             marker_color=col, text=[f"{v:.2f}" for v in vals],
                             textposition="outside"))
    fig.update_layout(title="Per-Intent Metrics", barmode="group",
                      yaxis=dict(range=[0,1.15], **_ax()), xaxis_tickangle=-30, **_L)
    fig.update_xaxes(**_ax())
    return fig

def radar(report: dict, classes: list):
    valid = [c for c in classes if c in report]
    scores = [report[c]["f1-score"] for c in valid]
    cats = valid + [valid[0]]; vals = scores + [scores[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                    line_color="#00C9A7", fillcolor="rgba(0,201,167,0.15)"))
    fig.update_layout(title="F1 Radar",
                      polar=dict(bgcolor=BG,
                                 radialaxis=dict(range=[0,1], gridcolor=GRID, color=FG),
                                 angularaxis=dict(gridcolor=GRID, color=FG)), **_L)
    return fig

def cv_bars(cv_acc: np.ndarray):
    folds = [f"Fold {i+1}" for i in range(len(cv_acc))]
    mean = cv_acc.mean()
    fig = go.Figure(go.Bar(x=folds, y=cv_acc,
                           marker_color=["#00C9A7" if s>=mean else "#FF6B6B" for s in cv_acc],
                           text=[f"{s:.3f}" for s in cv_acc], textposition="outside"))
    fig.add_hline(y=mean, line_dash="dash", line_color="#FFD166",
                  annotation_text=f"Mean {mean:.3f}", annotation_font_color="#FFD166")
    fig.update_layout(title="5-Fold CV Accuracy", yaxis=dict(range=[0,1.1], **_ax()), **_L)
    return fig

def conf_dist(pred_df: pd.DataFrame):
    ok  = pred_df[pred_df.correct]["confidence"].dropna()
    bad = pred_df[~pred_df.correct]["confidence"].dropna()
    fig = go.Figure()
    if len(ok):  fig.add_trace(go.Histogram(x=ok,  name="Correct",   marker_color="#00C9A7", opacity=0.75, nbinsx=20))
    if len(bad): fig.add_trace(go.Histogram(x=bad, name="Incorrect", marker_color="#FF6B6B", opacity=0.75, nbinsx=20))
    fig.update_layout(title="Confidence Distribution", barmode="overlay",
                      xaxis_title="Confidence", **_L)
    fig.update_xaxes(**_ax()); fig.update_yaxes(**_ax())
    return fig

def algo_compare(df: pd.DataFrame):
    fig = go.Figure()
    for i, metric in enumerate(["Accuracy","Precision","Recall","F1"]):
        if metric in df.columns:
            fig.add_trace(go.Bar(name=metric, x=df["Algorithm"], y=df[metric],
                                 marker_color=PALETTE[i],
                                 text=[f"{v:.3f}" for v in df[metric]], textposition="outside"))
    fig.update_layout(title="Algorithm Comparison", barmode="group",
                      yaxis=dict(range=[0,1.15], **_ax()), xaxis_tickangle=-20, **_L)
    return fig

def all_intent_scores(all_intents: list):
    intents = [x["intent"] for x in all_intents][::-1]
    confs   = [x["conf"]   for x in all_intents][::-1]
    max_c   = max(confs) if confs else 1
    colors  = ["#00C9A7" if c==max_c else "#1E2A3A" for c in confs]
    fig = go.Figure(go.Bar(x=confs, y=intents, orientation="h", marker_color=colors,
                           text=[f"{c*100:.1f}%" for c in confs], textposition="outside"))
    fig.update_layout(title="All Intent Scores", xaxis=dict(range=[0,1.2]),
                      height=max(200, len(intents)*42), **_L)
    fig.update_xaxes(**_ax())
    return fig
