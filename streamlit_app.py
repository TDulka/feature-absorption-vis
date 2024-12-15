import streamlit as st
import pandas as pd
import os
import random
import numpy as np
import plotly.graph_objects as go
import html
from streamlit_plotly_events import plotly_events
from plotly.subplots import make_subplots


st.set_page_config(layout="wide")

@st.cache_data
def load_available_sae_l0s():
    return pd.read_parquet("data/sae_split_feats.parquet")

@st.cache_data
def load_full_data():
    return pd.read_parquet("data/feature_absorption_results.parquet")

@st.cache_data
def load_sae_absorption_data(sae_l0, sae_width, layer):
    df = load_full_data()
    return df[
        (df["sae_l0"] == sae_l0)
        & (df["sae_width"] == sae_width)
        & (df["layer"] == layer)
        & (df["is_absorption"])
    ]

@st.cache_data
def load_english_tokens():
    return pd.read_parquet("data/english_tokens.parquet")

@st.cache_data
def get_random_letter_tokens(letter, n=30):
    tokens = load_english_tokens()
    letter_tokens = tokens[tokens["letter"] == letter]["token"].tolist()
    return random.sample(letter_tokens, min(n, len(letter_tokens)))

@st.cache_data
def get_random_non_letter_tokens(letter, n=30):
    tokens = load_english_tokens()
    letter_tokens = tokens[tokens["letter"] != letter]["token"].tolist()
    return random.sample(letter_tokens, n)

@st.cache_data
def get_sae_probe_cosine_similarities(sae_width, layer, sae_l0, letter):
    path = os.path.join(
        "data",
        "probe_sae_cos_sims",
        f"layer_{layer}",
        f"width_{sae_width}",
        f"l0_{sae_l0}",
        f"letter_{letter}.npz",
    )
    return np.load(path)["arr_0"].tolist()

@st.cache_data()
def load_top_feat_precision_recall():
    return pd.read_parquet("data/top_feat_precision_recall.parquet")
@st.cache_data
def load_k_sparse_probe_stats():
    return pd.read_parquet("data/k_sparse_results.parquet")

@st.cache_data
def load_html_dashboard(dashboard_url_or_path):
    with open(dashboard_url_or_path, "r") as file:
        dashboard_html = file.read()
    return dashboard_html


def get_probe_stats(layer, letter, sae_l0, sae_width):
    probe_stats = load_k_sparse_probe_stats()

    probe_stats = probe_stats[
        (probe_stats["layer"] == layer)
        & (probe_stats["letter"] == letter)
        & (probe_stats["sae_l0"] == sae_l0)
        & (probe_stats["sae_width"] == sae_width)
    ]

    return probe_stats


def is_canonical_sae(sae_width, layer, sae_l0):
    canonical_layer_l0_dict = {
        16000: {
            0: 105,
            1: 102,
            2: 141,
            3: 59,
            4: 124,
            5: 68,
            6: 70,
            7: 69,
            8: 71,
            9: 73,
            10: 77,
            11: 80,
            12: 82,
            13: 84,
            14: 84,
            15: 78,
            16: 78,
            17: 77,
            18: 74,
            19: 73,
            20: 71,
            21: 70,
            22: 72,
            23: 75,
            24: 73,
            25: 116,
        },
        65000: {
            0: 73,
            1: 121,
            2: 77,
            3: 89,
            4: 89,
            5: 105,
            6: 107,
            7: 107,
            8: 111,
            9: 118,
            10: 128,
            11: 70,
            12: 72,
            13: 75,
            14: 73,
            15: 127,
            16: 128,
            17: 125,
            18: 116,
            19: 115,
            20: 114,
            21: 111,
            22: 116,
            23: 123,
            24: 124,
            25: 93,
        },
    }

    return (
        sae_width in canonical_layer_l0_dict
        and layer in canonical_layer_l0_dict[sae_width]
        and canonical_layer_l0_dict[sae_width][layer] == sae_l0
    )

def get_dashboard_url_or_path(sae_width, layer, sae_l0, latent):
    if is_canonical_sae(sae_width, layer, sae_l0):
        sae_link_part = f"{layer}-gemmascope-res-{sae_width // 1000}k"
        return f"https://neuronpedia.org/gemma-2-2b/{sae_link_part}/{latent}?embed=true"
    else:
        return os.path.join(
            "data",
            "non_canonical_dashboards",
            f"layer_{layer}",
            f"width_{sae_width // 1000}k",
            f"average_l0_{sae_l0}_feature_{latent}.html",
        )

def display_dashboard(sae_width, layer, sae_l0, latent):
    dashboard_url_or_path = get_dashboard_url_or_path(sae_width, layer, sae_l0, latent)
    
    if is_canonical_sae(sae_width, layer, sae_l0):
        iframe_html = f"""
        <iframe src="{dashboard_url_or_path}" class="stIFrame" style="border:none; width:100%;" height="800" loading="lazy" scrolling="yes"></iframe>
        """

        st.components.v1.html(iframe_html, height=800, scrolling=True)
    else:
        try:
            dashboard_html = load_html_dashboard(dashboard_url_or_path)

            css_modification = """
            .grid-container {
                display: flex;
                flex-direction: column;
                margin: 0;
                padding-left: 0;
                padding-top: 20px;
                white-space: wrap;
                overflow-x: none;
                box-sizing: border-box;
            }
            .grid-column {
                max-height: none !important;
                width: 100%;
                box-sizing: border-box;
                margin: 0;
                padding: 0 20px;
            }
            div.logits-table {
                min-width: 0px;
                flex-wrap: wrap;
            }
            div.logits-table > div.negative {
                width: auto;
                flex: 1;
            }
            div.logits-table > div.positive {
                width: auto;
                flex: 1;
            }
            #column-0 {
                display: none;
            }
            """
            # Insert the CSS modification just before the closing </style> tag
            modified_html = dashboard_html.replace(
                "</style>", f"{css_modification}</style>"
            )

            # Properly escape the modified_html for use in srcdoc
            escaped_html = html.escape(modified_html, quote=True)

            iframe_html = f"""
            <iframe class='stIFrame' width='100%' height='800' loading='lazy' scrolling='yes' 
            style="border:none; width:100%;"
                    srcdoc="{escaped_html}">
            </iframe>
            """

            st.components.v1.html(iframe_html, height=900, scrolling=True)
        except FileNotFoundError:
            st.error(
                f"Dashboard for latent {latent} not found. This may be due to the file being missing."
            )



def plot_sae_probe_cosine_similarities(similarities, split_latents, absorbing_latents):
    # Define a color theme
    color_theme = {
        "background": "white",
        "grid": "lightgrey",
        "line": "#CCCCCC",  # Light grey for the main line
        "split": "#1f77b4",  # Blue for split latents
        "absorbing": "#ff7f0e",  # Orange for absorbing latents
    }

    fig = go.Figure()

    # Plot all similarities in light gray
    fig.add_trace(
        go.Scatter(
            y=similarities,
            mode="lines",
            line=dict(color=color_theme["line"]),
            name="Cosine Similarity",
        )
    )

    # Highlight split latents in blue
    split_x = [i for i in range(len(similarities)) if i in split_latents]
    split_y = [similarities[i] for i in split_x]
    fig.add_trace(
        go.Scatter(
            x=split_x,
            y=split_y,
            mode="markers",
            marker=dict(color=color_theme["split"], size=9, symbol="diamond"),
            name="Split Latents",
        )
    )

    # Highlight absorbing latents in orange
    absorption_x = [i for i in range(len(similarities)) if i in absorbing_latents]
    absorption_y = [similarities[i] for i in absorption_x]
    fig.add_trace(
        go.Scatter(
            x=absorption_x,
            y=absorption_y,
            mode="markers",
            marker=dict(color=color_theme["absorbing"], size=9),
            name="Absorbing Latents",
        )
    )

    y_min = min(-0.3, min(similarities) - 0.1)
    y_max = max(0.7, max(similarities) + 0.1)

    fig.update_layout(
        title=dict(
            text="SAE Latents & Linear Probe Cosine Similarities", font=dict(size=24)
        ),
        xaxis_title="Latent Index",
        yaxis_title="Cosine Similarity",
        height=400,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor=color_theme["background"],
        paper_bgcolor=color_theme["background"],
        xaxis=dict(
            showgrid=True,
            gridcolor=color_theme["grid"],
            gridwidth=1,
            zeroline=True,
            zerolinecolor=color_theme["grid"],
            zerolinewidth=1,
        ),
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=True,
            gridcolor=color_theme["grid"],
            gridwidth=1,
            zeroline=True,
            zerolinecolor=color_theme["grid"],
            zerolinewidth=1,
        ),
    )

    fig.update_traces(
        hoverinfo="text",
        hovertemplate="<b>Latent:</b> %{x:.0f}<br><b>Cosine Similarity:</b> %{y:.4f}<extra></extra>",
    )

    return fig

def plot_k_sparse_f1_scores(probe_stats):
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50]
    f1_scores = [probe_stats[f"f1_sparse_sae_{k}"].iloc[0] for k in k_values]
    split_feats = probe_stats["split_feats"].iloc[0]
    num_split_feats = len(split_feats)

    # Define colors
    bar_colors = [
        "#1f77b4" if i < num_split_feats else "#7f7f7f" for i in range(len(k_values))
    ]

    # Create labels with k values and split feature numbers
    labels = [
        f"{k} ({'+ ' if i > 0 else ''}{split_feats[i]})"
        if i < num_split_feats
        else str(k)
        for i, k in enumerate(k_values)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(k) for k in k_values],  # Convert k values to strings
                y=f1_scores,
                marker_color=bar_colors,
            )
        ]
    )

    fig.update_layout(
        title=dict(text="F1 Scores for k-Sparse Probes", font=dict(size=24)),
        xaxis_title="k (Split Latent Number)",
        yaxis_title="F1 Score",
        height=400,
        yaxis=dict(range=[0, 1]),
        xaxis=dict(
            type="category",  # Set x-axis type to category
            categoryorder="array",
            categoryarray=[str(k) for k in k_values],  # Ensure correct order
            tickangle=45,  # Rotate labels by 45 degrees
            tickmode="array",
            tickvals=[str(k) for k in k_values],
            ticktext=labels,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add gridlines
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
        zeroline=True,
        zerolinecolor="lightgrey",
        zerolinewidth=1,
    )

    return fig

def plot_combined_precision_recall(df):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Precision vs L0", "Recall vs L0", "Precision vs Recall"),
    )

    # Precision vs L0
    colors = ["#1f77b4", "#ff7f0e"]
    width_color_map = {16000: colors[0], 65000: colors[1]}
    fig.add_trace(
        go.Scatter(
            x=df["sae_l0"],
            y=df["precision_sae_top_0"],
            mode="markers",
            marker=dict(
                size=8,
                color=[width_color_map[width] for width in df["sae_width"]],
            ),
            text=[
                f"Layer: {layer}<br>Width: {width}<br>L0: {l0}"
                for layer, width, l0 in zip(df["layer"], df["sae_width"], df["sae_l0"])
            ],
            hoverinfo="text+y",
        ),
        row=1,
        col=1,
    )

    # Recall vs L0
    fig.add_trace(
        go.Scatter(
            x=df["sae_l0"],
            y=df["recall_sae_top_0"],
            mode="markers",
            marker=dict(
                size=8,
                color=[width_color_map[width] for width in df["sae_width"]],
            ),
            text=[
                f"Layer: {layer}<br>Width: {width}<br>L0: {l0}"
                for layer, width, l0 in zip(df["layer"], df["sae_width"], df["sae_l0"])
            ],
            hoverinfo="text+y",
        ),
        row=1,
        col=2,
    )

    # Precision vs Recall
    min_l0 = df["sae_l0"].min()
    max_l0 = df["sae_l0"].max()

    def norm(x):
        return ((x - min_l0) / (max_l0 - min_l0)) ** 0.5

    fig.add_trace(
        go.Scatter(
            x=df["precision_sae_top_0"],
            y=df["recall_sae_top_0"],
            mode="markers",
            marker=dict(
                size=8,
                color=[norm(l0) for l0 in df["sae_l0"]],
                colorscale="cividis",
                colorbar=dict(
                    title=dict(text="L0", side="right"),
                    tickvals=[0, 0.5, 1],
                    ticktext=[
                        f"{min_l0:.0f}",
                        f"{((min_l0 + max_l0) / 2):.0f}",
                        f"{max_l0:.0f}",
                    ],
                ),
                showscale=True,
            ),
            text=[
                f"Layer: {layer}<br>Width: {width}<br>L0: {l0}"
                for layer, width, l0 in zip(df["layer"], df["sae_width"], df["sae_l0"])
            ],
            hoverinfo="text",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=dict(
            text="Main SAE Latent Precision and Recall Plots", font=dict(size=24)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        hovermode="closest",
    )

    return fig


def initialize_tasks():
    if "tasks" not in st.session_state:
        st.session_state.tasks = [
            {
                "id": "select_letter",
                "description": 'Select a letter (e.g., "L") and observe its split latents',
                "hint": 'Use the sidebar to select a letter and look at the "Split latents" section',
                "completed": False,
            },
            {
                "id": "compare_metrics",
                "description": "Compare the performance of the main SAE latent vs. the Linear Probe",
                "hint": 'Check the "Comparison of main SAE split latent and Linear Probe performance" section',
                "completed": False,
            },
            {
                "id": "test_split_latent",
                "description": "Convince yourself that the split latent is capturing the feature of interest by testing it on random words starting with the selected letter",
                "hint": "Click on a split latent and use the embedded Neuronpedia dashboard to test activations",
                "completed": False,
            },
            {
                "id": "find_non_activating",
                "description": "Find words that start with the selected letter but don't strongly activate the main split latent",
                "hint": "You can copy tokens from the Absorbing Latents section or come up with your own.",
                "completed": False,
            },
            {
                "id": "test_absorbing_latent",
                "description": "Identify an absorbing latent and test its tokens using Neuronpedia",
                "hint": 'Look at the "Absorbing Latents" section and use the embedded Neuronpedia dashboard',
                "completed": False,
            },
            {
                "id": "compare_activations",
                "description": "Compare activations of absorbed tokens on the absorbing latent vs. the main split latent",
                "hint": "Use Neuronpedia to test the same tokens on both the absorbing and main split latents",
                "completed": False,
            },
            {
                "id": "explore_cosine_similarities",
                "description": "Explore the cosine similarities graph and click on different latents",
                "hint": 'Check out the "Cosine Similarities" section and interact with the graph',
                "completed": False,
            },
            {
                "id": "compare_letters",
                "description": "Repeat the process for a different letter and compare the feature absorption behavior",
                "hint": "Select a new letter from the sidebar and go through the previous steps again",
                "completed": False,
            },
            {
                "id": "investigate_canonical",
                "description": "Switch between a canonical and non-canonical SAE for the same letter",
                "hint": 'Use the "Select SAE L0" dropdown in the sidebar to switch between SAEs',
                "completed": False,
            },
            {
                "id": "analyze_k_sparse",
                "description": "Analyze the k-sparse probe graph to understand how F1 score changes",
                "hint": 'Expand the "How we calculate feature splitting" section and examine the graph',
                "completed": False,
            },
        ]


def render_task_list():
    st.sidebar.markdown("---")
    with st.sidebar.expander("Feature Absorption Discovery Tasks", expanded=True):
        st.write("Complete these tasks to explore feature absorption:")

        for task in st.session_state.tasks:
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                task["completed"] = st.checkbox(
                    task["id"],
                    key=f"task_{task['id']}",
                    value=task["completed"],
                    label_visibility="collapsed",
                )
            with col2:
                if task["completed"]:
                    st.markdown(f"~~{task['description']}~~")
                else:
                    st.write(task["description"])
                    st.info(task["hint"])

        if st.button("Reset Tasks"):
            for task in st.session_state.tasks:
                task["completed"] = False

def main():
    hide_elements = """
    <style>
    header {visibility: hidden;}
    [data-testid="manage-app-button"] {display: none;}
    </style>
    """
    st.markdown(hide_elements, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Feature Absorption Explorer", "Cosine Similarity Comparisons"]
    )

    if page == "Feature Absorption Explorer":
        feature_absorption_explorer()
    elif page == "Cosine Similarity Comparisons":
        cosine_similarity_comparison()


def cosine_similarity_comparison():
    st.title("SAE Latent Cosine Similarities With Linear Probe")

    st.write(
        "These plots allow you to compare cosine similarities across different L0 values for each SAE width. "
        "You can use the sidebar to select different layers and letters for comparison."
    )

    available_saes_df = load_available_sae_l0s()

    # Sidebar selectors
    st.sidebar.subheader("Select parameters for comparison")

    layers = sorted(available_saes_df["layer"].unique())
    selected_layer = st.sidebar.selectbox("Select Layer", layers)

    letters = sorted(available_saes_df["letter"].unique())
    selected_letter = st.sidebar.selectbox("Select Letter", letters)

    # Function to plot cosine similarities for a given width
    def plot_cosine_similarities(width):
        available_l0s = sorted(
            available_saes_df[
                (available_saes_df["layer"] == selected_layer)
                & (available_saes_df["sae_width"] == width)
            ]["sae_l0"].unique()
        )

        for l0 in available_l0s:
            similarities = get_sae_probe_cosine_similarities(
                width, selected_layer, l0, selected_letter
            )

            # Get split latents using get_probe_stats()
            probe_stats = get_probe_stats(selected_layer, selected_letter, l0, width)
            split_latents = (
                probe_stats["split_feats"].iloc[0] if not probe_stats.empty else []
            )

            absorption_data = load_sae_absorption_data(l0, width, selected_layer)
            absorbing_latents = absorption_data[
                (absorption_data["letter"] == selected_letter)
            ]["ablation_feat"].unique()

            fig = plot_sae_probe_cosine_similarities(
                similarities, split_latents, absorbing_latents
            )
            fig.update_layout(
                title=f"Cosine Similarities for Width {width}, L0 {l0}, Layer {selected_layer}, Letter {selected_letter}"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Plot for width 16000
    st.subheader("Cosine Similarities for SAE Width 16000")
    plot_cosine_similarities(16000)

    # Plot for width 65000
    st.subheader("Cosine Similarities for SAE Width 65000")
    plot_cosine_similarities(65000)


def feature_absorption_explorer():
    st.title("Feature Absorption Results Explorer")

    available_saes_df = load_available_sae_l0s()

    # Get query parameters
    query_params = st.query_params

    # Move selectors to the sidebar
    st.sidebar.subheader("Select an SAE and the first letter to explore")

    layers = sorted(available_saes_df["layer"].unique())
    default_layer = int(query_params.get("layer", layers[0]))
    selected_layer = st.sidebar.selectbox(
        "Select Layer",
        layers,
        key="layer",
        index=layers.index(default_layer) if default_layer in layers else 0,
    )

    sae_widths = sorted(available_saes_df["sae_width"].unique())
    default_sae_width = int(query_params.get("sae_width", sae_widths[0]))
    selected_sae_width = st.sidebar.selectbox(
        "Select SAE Width",
        sae_widths,
        key="sae_width",
        index=sae_widths.index(default_sae_width)
        if default_sae_width in sae_widths
        else 0,
    )

    filtered_df = available_saes_df[
        (available_saes_df["layer"] == selected_layer)
        & (available_saes_df["sae_width"] == selected_sae_width)
    ]
    available_l0s = sorted(filtered_df["sae_l0"].unique())

    # Find the canonical L0 for the selected layer and width
    canonical_l0 = next(
        (
            l0
            for l0 in available_l0s
            if is_canonical_sae(selected_sae_width, selected_layer, l0)
        ),
        available_l0s[0],  # Default to the first L0 if no canonical is found
    )

    default_sae_l0 = int(query_params.get("sae_l0", canonical_l0))

    selected_sae_l0 = st.sidebar.selectbox(
        "Select SAE L0",
        available_l0s,
        index=available_l0s.index(default_sae_l0)
        if default_sae_l0 in available_l0s
        else available_l0s.index(canonical_l0),
        key="sae_l0",
    )

    # Highlight if the selected SAE is canonical
    is_canonical = is_canonical_sae(selected_sae_width, selected_layer, selected_sae_l0)
    if is_canonical:
        st.sidebar.success("Selected SAE is canonical (on Neuronpedia)")
    else:
        st.sidebar.info("Selected SAE is non-canonical (not on Neuronpedia)")

    available_letters = filtered_df[filtered_df["sae_l0"] == selected_sae_l0][
        "letter"
    ].unique()

    # Count absorbing latents for each letter
    absorption_data = load_sae_absorption_data(
        selected_sae_l0, selected_sae_width, selected_layer
    )
    letter_absorbing_latents = {}
    for letter in available_letters:
        absorbing_latents = absorption_data[(absorption_data["letter"] == letter)][
            "ablation_feat"
        ].nunique()
        letter_absorbing_latents[letter] = absorbing_latents

    # Create letter options with absorbing latent counts
    letter_options = [
        f"{letter} ({letter_absorbing_latents[letter]})" for letter in available_letters
    ]

    default_letter = query_params.get("letter", available_letters[0])
    selected_letter_option = st.sidebar.selectbox(
        "Select Letter (count of available absorbing latents in parentheses)",
        letter_options,
        index=available_letters.tolist().index(default_letter)
        if default_letter in available_letters
        else 0,
        key="letter",
    )

    # Extract the letter from the selected option
    selected_letter = selected_letter_option.split()[0]

    # Update query parameters
    new_query_params = {
        "layer": selected_layer,
        "sae_width": selected_sae_width,
        "sae_l0": selected_sae_l0,
        "letter": selected_letter,
    }
    st.query_params.update(new_query_params)

    # Store the selected letter in session state
    st.session_state.selected_letter = selected_letter

    initialize_tasks()
    render_task_list()

    final_df = filtered_df[
        (filtered_df["sae_l0"] == selected_sae_l0)
        & (filtered_df["letter"] == selected_letter)
    ]

    result_df = (
        final_df.groupby("letter")
        .agg(
            {
                "num_true_positives": "first",
                "split_feats": "first",
            }
        )
        .reset_index()
    )

    letter_absorptions = absorption_data[absorption_data["letter"] == selected_letter]

    latent_tokens = (
        letter_absorptions.groupby("ablation_feat")["token"].apply(list).reset_index()
    )

    latent_unique_tokens = {}

    for _, row in latent_tokens.iterrows():
        latent = row["ablation_feat"]
        tokens = row["token"]
        unique_tokens = list(set(tokens))  # Remove duplicates
        latent_unique_tokens[latent] = unique_tokens

    sae_probe_cosine_similarities = get_sae_probe_cosine_similarities(
        selected_sae_width, selected_layer, selected_sae_l0, selected_letter
    )

    # Get split latents
    split_latents = result_df[result_df["letter"] == selected_letter][
        "split_feats"
    ].iloc[0]

    # Get absorbing latents
    absorbing_latents = letter_absorptions["ablation_feat"].unique()

    probe_stats = get_probe_stats(
        selected_layer, selected_letter, selected_sae_l0, selected_sae_width
    )

    with st.expander("What is feature absorption?", expanded=True):
        st.write(
            'This app demonstrates a particularly problematic case of feature splitting we call "feature absorption" where a seemingly interpretable monosemantic latent '
            'capturing a feature like "first letter is L" has many exceptions captured by other latents.'
            # 'Our paper with full analysis can be found here: https://arxiv.org/abs/2409.14507.'
        )

        st.write(
            'When working with Sparse Autoencoders (SAEs), you might expect that when you find an SAE latent capturing a feature like "first letter of token is L", '
            "it will be good at distinguishing tokens starting with L from those that don't. If the latent isn't a great classifier, "
            "you might think it's only because the feature is split into multiple latents in this particular SAE, perhaps one for lowercase L and one for uppercase L and you can find those latents. "
            "You might suppose that there will be a certain width and sparsity of your SAE where the feature splits into a handful of interpretable latents."
        )

        st.write(
            "However, we attempt to demonstrate that you'll likely encounter a more problematic behavior called feature absorption. You might indeed find a couple of latents that seem to be the main "
            '"first letter is L" latents capturing many tokens starting with L, but they will have seemingly random exceptions, e.g. "_legal", "_load", "_longtime", and others. '
            'For these exception tokens, a different set of SAE latents will absorb the "first letter is L" direction. These absorbing latents would be very hard to discover without the ground truth data.'
        )

        st.write(
            "This app aims to demonstrate that feature absorption is a phenomenon that occurs and should be considered when interpreting SAE latents. "
            "Our metrics for classifying where feature splitting and feature absorption happen are imperfect, so we don't claim the results are exhaustive. "
            "Consider them as an existence proof of a problematic behavior."
        )

    with st.expander("How we calculate feature **splitting**"):
        st.write(
            "We measure feature splitting using k-sparse probing on SAE activations. "
            "This method involves training a logistic regression probe on the top k SAE latents "
            "that are most predictive of the first-letter task. A significant increase in the "
            "probe's F1 score when moving from k to k+1 latents indicates that the additional "
            "latent provides meaningful signal, suggesting a feature split."
        )
        st.write(
            "For example, in the case of a split between capital 'L' and lowercase 'l' features, "
            "a k-sparse probe with k=2 trained on both these features would likely predict "
            "'starts with letter L' much better than either feature alone. This improvement "
            "in prediction accuracy is indicative of feature splitting."
        )
        st.write(
            "The effectiveness of this method can be visualized by plotting F1 score against k. "
            "For instance, the k-sparse probe for the letter 'L' might show a significant jump "
            "in F1 score when moving from k=1 to k=2, corresponding to feature splitting. In contrast, "
            "for a letter like 'N' where splitting might not occur, the F1 score could remain "
            "relatively constant across different k values."
        )
        st.write(
            "We detect feature splitting by measuring whether increasing k by one causes a jump in F1 score "
            "by more than a threshold tau. We set tau to 0.03 after manually inspecting features with "
            "various thresholds. You can see this visually in a figure below."
        )

        if not probe_stats.empty:
            fig_k_sparse = plot_k_sparse_f1_scores(probe_stats)
            st.plotly_chart(fig_k_sparse, use_container_width=True)

            st.write(
                "The blue bars represent which latents we categorize as split, with their corresponding numbers shown in parentheses."
            )

    with st.expander("How we determine feature **absorption**"):
        st.write(
            "We determine whether feature absorption has occurred for a particular latent through the following process:"
        )
        st.write(
            "1. We first identify k feature splits for the given first-letter latent using a k-sparse probe."
        )
        st.write(
            "2. We then find false-negative tokens that all k feature-split SAE latents fail to activate on, but which a linear probe correctly classifies."
        )
        st.write(
            "3. For these tokens, we run an integrated-gradients ablation experiment to find the most causally important SAE latents for the spelling of that token."
        )
        st.write(
            "4. We consider feature absorption to have occurred if the SAE latent receiving the largest negative magnitude ablation effect has a cosine similarity with the linear probe above 0.025, and its ablation effect is larger by at least 1.0 than the second highest ablation effect."
        )
        st.write(
            "It's important to note that this approach may not capture all instances of feature absorption, such as cases where multiple latents absorb the feature together or where the main latents continue to activate but very weakly."
        )

    if not probe_stats.empty:
        precision_probe = probe_stats["precision_probe"].iloc[0]
        recall_probe = probe_stats["recall_probe"].iloc[0]
        f1_probe = probe_stats["f1_probe"].iloc[0]

        precision_sae = probe_stats["precision_sparse_sae_1"].iloc[0]
        recall_sae = probe_stats["recall_sparse_sae_1"].iloc[0]
        f1_sae = probe_stats["f1_sparse_sae_1"].iloc[0]

        top_sae = probe_stats["split_feats"].iloc[0][0]

        st.subheader(
            "Comparison of main SAE split latent and Linear Probe classification performance"
        )

        with st.expander(
            "You can compare the precision and recall of the main SAE latent averaged across all letters, for all SAE widths and L0s.",
            expanded=False,
        ):
            pr_data = load_top_feat_precision_recall()
            pr_data = (
                pr_data.groupby(["sae_width", "sae_l0", "layer"])
                .agg({"precision_sae_top_0": "mean", "recall_sae_top_0": "mean"})
                .reset_index()
            )

            combined_fig = plot_combined_precision_recall(pr_data)
            st.plotly_chart(combined_fig, use_container_width=True)

        st.write(
            f"Here we show the comparison of classification performance when using the main SAE latent ({top_sae}) from SAE width {selected_sae_width} and SAE L0 {selected_sae_l0} with the linear probe at predicting first letter '{selected_letter}' (ignoring case) from model's activation at layer {selected_layer}:"
        )

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7, gap="small")

        col1.metric("SAE Precision", f"{precision_sae:.3f}")
        col2.metric("SAE Recall", f"{recall_sae:.3f}")
        col3.metric("SAE F1 Score", f"{f1_sae:.3f}")

        col5.metric("Linear Probe Precision", f"{precision_probe:.3f}")
        col6.metric("Linear Probe Recall", f"{recall_probe:.3f}")
        col7.metric("Linear Probe F1 Score", f"{f1_probe:.3f}")

        st.subheader("Cosine Similarities")

        st.write(
            "We observe that in most cases, the SAE latents that we categorize as split based on k-sparse probing also have a high cosine similarity to the linear probe. "
            "Note we only test absorption on 20% of the vocabulary (the test set of linear probes we train) so not all absorbing latents will be shown on this plot and below."
        )

        fig_cosine = plot_sae_probe_cosine_similarities(
            sae_probe_cosine_similarities, split_latents, absorbing_latents
        )
        selected_points = plotly_events(fig_cosine, click_event=True)

        if selected_points:
            clicked_latent = int(selected_points[0]["x"])
            if is_canonical_sae(selected_sae_width, selected_layer, selected_sae_l0):
                sae_link_part = (
                    f"{selected_layer}-gemmascope-res-{selected_sae_width // 1000}k"
                )
                neuronpedia_url = f"https://neuronpedia.org/gemma-2-2b/{sae_link_part}/{clicked_latent}?embed=true"
                with st.expander(
                    f"View Neuronpedia dashboard for latent {clicked_latent}",
                    expanded=True,
                ):
                    st.components.v1.iframe(neuronpedia_url, height=600, scrolling=True)
            else:
                st.write(
                    f"Selected latent {clicked_latent} for non-canonical SAE is not available on Neuronpedia."
                )
        elif is_canonical:
            st.write("Click on any latent on the plot to see its neuronpedia page.")

    selected_letter_latents = result_df[result_df["letter"] == selected_letter][
        "split_feats"
    ].iloc[0]

    st.header(
        f"Latents in the selected SAE associated with the feature 'first letter is {selected_letter}'"
    )

    left_column, right_column = st.columns(2)

    n_dashboards_to_display = 20

    with left_column:
        st.subheader(f"Split latents ({len(selected_letter_latents)})")

        if is_canonical:
            latents_str = ", ".join([str(latent) for latent in selected_letter_latents])

            latent_str = "latent" if len(selected_letter_latents) == 1 else "latents"

            st.write(
                f"The {latent_str} {latents_str} should be the primary 'first letter is {selected_letter}' {latent_str}.",
                f"You should be able to test the activation with random words starting with letter {selected_letter} below.",
                f"\n\nTry finding words that start with {selected_letter} that don't activate the latent.",
                "You can compare them with the tokens we have discovered in the right column.",
            )

    with right_column:
        st.subheader(f"Absorbing Latents ({len(latent_unique_tokens)})")

        if not latent_unique_tokens:
            st.write("No absorbing latents found for this selection.")
        else:
            all_unique_tokens = set()
            for tokens in latent_unique_tokens.values():
                all_unique_tokens.update(tokens)

            all_unique_tokens = ",".join(list(all_unique_tokens))

            if is_canonical:
                st.write(
                    f"We have discovered that some latents capture the 'first letter is {selected_letter}' signal on specific tokens. "
                    "Try copying the tokens showing absorption and test their activations on the main latent and compare with the absorbing latents."
                )

        if len(latent_unique_tokens) > n_dashboards_to_display:
            st.write(
                f"Displaying only the first {n_dashboards_to_display} absorbing latents for performance reasons."
            )

    left_column_iframe, right_column_iframe = st.columns(2)

    with left_column_iframe:
        latent_tabs = st.tabs(
            [f"Latent: {latent}" for latent in selected_letter_latents]
        )

        for tab, latent in zip(latent_tabs, selected_letter_latents):
            with tab:
                if is_canonical:
                    st.write(
                        f"Random '{selected_letter}' tokens from the vocab for testing:"
                    )
                    st.code(f"{','.join(get_random_letter_tokens(selected_letter))}")

                    st.write(
                        f"Random non-{selected_letter} tokens from the vocab for testing:"
                    )
                    st.code(
                        f"{','.join(get_random_non_letter_tokens(selected_letter))}"
                    )

                display_dashboard(
                    selected_sae_width, selected_layer, selected_sae_l0, latent
                )

    with right_column_iframe:
        if len(latent_unique_tokens) > 0:
            latent_unique_tokens_capped = dict(
                list(latent_unique_tokens.items())[:n_dashboards_to_display]
            )

            latent_tabs = st.tabs(
                [
                    f"Latent: {latent} ({', '.join(tokens)})"
                    for latent, tokens in latent_unique_tokens_capped.items()
                ]
            )

            for i, (tab, (latent, tokens)) in enumerate(
                zip(latent_tabs, latent_unique_tokens_capped.items())
            ):
                with tab:
                    if is_canonical:
                        st.write(f"Tokens absorbed by {latent}:")
                        st.code(f"{','.join(tokens)}")

                        st.write("Tokens across all absorbing latents:")
                        st.code(all_unique_tokens)

                    display_dashboard(
                        selected_sae_width, selected_layer, selected_sae_l0, latent
                    )


if __name__ == "__main__":
    main()
