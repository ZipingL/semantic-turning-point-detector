import json, pandas as pd, plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------------
# A)  LOAD THE DIALOGUE LINES  (unchanged, keeps nice hover snippets)
# ------------------------------------------------------------------
conv_path = Path("src/conversationPariah.json")
if conv_path.exists():
    with open(conv_path, "r", encoding="utf-8") as f:
        convo_raw = json.load(f)
    convo_df = pd.json_normalize(convo_raw).rename(columns={"id": "index"})
    line_lookup = {r.index: f"{r.author}: {r.message[:140]}…" for r in convo_df.itertuples()}
else:
    line_lookup = {}

# ------------------------------------------------------------------
# B)  TURNING-POINT DATA  – now with tone, sig, quote  ─────────────
#     (I shortened quotes to keep the snippet small – expand freely)
# ------------------------------------------------------------------
model_data = {
    "Qwen 3-1.7 B": dict(
        indices  =[0,0,87,96,99,104,108,110,112,114,119,127,130,134,141,150],
        categories=["Emotion","Meta-Reflection","Meta-Reflection","Meta-Reflection",
                    "Meta-Reflection","Emotion","Meta-Reflection","Emotion",
                    "Meta-Reflection","Meta-Reflection","Emotion","Meta-Reflection",
                    "Emotion","Meta-Reflection","Emotion","Meta-Reflection"],
        tones     =["angry","angry","angry","angry","angry","angry","anxious",
                    "angry","anxious","anxious","angry","angry","angry",
                    "furious","angry","angry"],
        sig       =[1.00]*16,
        quotes    =["You – did?","You – did?","You jest!","Yes.","No. An accident…",
                    "So you think …","Well, we understand…","It is to me that …",
                    "No? Yes, you have me.","I don’t want to become a thief.",
                    "I see in the mirror…","Now everything is clear …","Through need…",
                    "[Completely defeated] May I go now?","Shall we have another bout?",
                    "You were too cowardly… Can I go?"],
        processing_time="03:51", avg_significance=.376, count=16
    ),
    "GPT-4.1-nano": dict(
        indices  =[0,0,11,11,122,127,142,144,150,152],
        categories=["Emotion","Clarification","Emotion","Other","Insight","Other",
                    "Insight","Decision","Objection","Question"],
        tones     =["skeptical"]*4+["skeptical","skeptical","skeptical",
                    "skeptical","angry","angry"],
        sig       =[1.00,1.00,1.00,1.00,1.00,1.00,.93,.93,1.00,1.00],
        quotes    =["What for?","What for?","Do you think I would give …",
                    "I should not be able …","You are pretty crafty …",
                    "Shift from authority …","that is my secret …",
                    "Yes, and you cannot prevent it.","You were too cowardly …",
                    "Can I go?"],
        processing_time="03:21", avg_significance=.342, count=10
    ),
    #  ➜ GPT-4.1 (full), Qwen 30 B, Gemini Flash … fill in the same way
}

# ------------------------------------------------------------------
# C)  TIDY RECORDS  (now include tone, sig, quote)  ────────────────
# ------------------------------------------------------------------
records = []
for mdl, d in model_data.items():
    for i, idx in enumerate(d["indices"]):
        records.append(dict(
            model=mdl,
            index=idx,
            category=d["categories"][i],
            tone=d["tones"][i],
            sig=d["sig"][i],
            quote=d["quotes"][i],
            snippet=line_lookup.get(idx, "(line not found)")
        ))
df = pd.DataFrame(records)

# ------------------------------------------------------------------
# D)  FIGURE  – marker size ∝ significance, hover shows everything
# ------------------------------------------------------------------
category_colors = {       # same palette; colour by category
    "Meta-Reflection":"#FF6B6B","Emotion":"#4ECDC4","Decision":"#45B7D1",
    "Insight":"#96CEB4","Action":"#FFEAA7","Question":"#DDA0DD","Problem":"#FFB347",
    "Topic":"#98FB98","Conflict":"#F0E68C","Objection":"#FFA07A","Clarification":"#87CEFA",
    "Other":"#B0C4DE","Unknown":"#D3D3D3"
}
# ── variant: colour by tone instead → swap the dict & use tone column

fig = go.Figure()
for mdl in model_data:
    mdf = df[df.model == mdl]
    for cat in mdf.category.unique():
        cdf = mdf[mdf.category == cat]
        fig.add_trace(go.Scatter(
            x=cdf.index, y=[mdl]*len(cdf),
            mode="markers",
            marker=dict(
                size=[8+sig*10 for sig in cdf.sig],   # 8–18 px
                color=[category_colors.get(cat,"#D3D3D3")]*len(cdf),
                line=dict(width=1.2,color="white")),
            name=cat, legendgroup=cat,
            showlegend=cat not in [t.name for t in fig.data],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Idx: %{x}<br>"
                "Class: "+cat+"<br>"
                "Tone: %{customdata[0]}<br>"
                "Significance: %{customdata[1]:.2f}<br>"
                "Quote: %{customdata[2]}<extra></extra>"
            ),
            customdata=cdf[["tone","sig","quote"]].values
        ))

# ---------- convergence lines etc. (unchanged) ----------
conv = df.index.value_counts().reset_index(name="ct")
conv_pts = conv[conv.ct >= 3].sort_values("index")
for _, r in conv_pts.iterrows():
    fig.add_vline(x=r["index"],
                  line=dict(color="rgba(128,128,128,0.4)", width=3, dash="dash"),
                  annotation_text=f"{r.ct} models",
                  annotation_position="top",
                  annotation=dict(font=dict(size=11,color="darkblue")))

fig.update_layout(
    title=("Semantic Turning Points • Strindberg ‘Pariah’"
           "<br><sub>size ∝ significance | colour = semantic class</sub>"),
    xaxis_title="Dialogue message index", yaxis_title="Model",
    width=1600, height=830, plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=220,r=40,t=110,b=140),
    legend=dict(orientation="v",x=1.02,y=1,title="Category")
)
fig.update_xaxes(showgrid=True,gridcolor="rgba(200,200,200,.3)",range=[-5,160])
fig.update_yaxes(showgrid=True,gridcolor="rgba(200,200,200,.3)",
                 categoryorder="array",
                 categoryarray=list(reversed(list(model_data))))
fig.show()

# ------------------------------------------------------------------
# E)  quick convergence printout (kept)  ───────────────────────────
# ------------------------------------------------------------------
print("\nKEY CONVERGENCE POINTS (≥3 models)")
for _, r in conv_pts.iterrows():
    idx=r["index"]
    involved=", ".join(df[df.index==idx].model.unique())
    print(f"Idx {idx:3d} | {r.ct} models | {involved}")
