import streamlit as st
import pandas as pd
import os
import random
import numpy as np
import plotly.graph_objects as go
from urllib.parse import urlencode

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
        & (df["feat_order"] == 0)
        & (df["is_absorption"])
    ]

@st.cache_data
def load_english_tokens():
    return pd.read_parquet("data/english_tokens.parquet")


def get_random_letter_tokens(letter, n=30):
    tokens = load_english_tokens()
    letter_tokens = tokens[tokens["letter"] == letter]["token"].tolist()
    return random.sample(letter_tokens, min(n, len(letter_tokens)))


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


@st.cache_data
def load_probe_stats():
    return pd.read_parquet("data/probe_stats_across_layers.parquet")


def get_probe_stats(layer, letter):
    probe_stats = load_probe_stats()
    return probe_stats[
        (probe_stats["layer"] == layer) & (probe_stats["letter"] == letter)
    ]

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
        },
    }

    return (
        sae_width in canonical_layer_l0_dict
        and layer in canonical_layer_l0_dict[sae_width]
        and canonical_layer_l0_dict[sae_width][layer] == sae_l0
    )

def get_dashboard_url_or_path(sae_width, layer, sae_l0, feature):
    if is_canonical_sae(sae_width, layer, sae_l0):
        sae_link_part = f"{layer}-gemmascope-res-{sae_width // 1000}k"
        return f"https://neuronpedia.org/gemma-2-2b/{sae_link_part}/{feature}?embed=true"
    else:
        return os.path.join(
            "data",
            "non_canonical_dashboards",
            f"layer_{layer}",
            f"width_{sae_width // 1000}k",
            f"average_l0_{sae_l0}_feature_{feature}.html"
        )
    
def display_dashboard(sae_width, layer, sae_l0, feature):
    dashboard_url_or_path = get_dashboard_url_or_path(sae_width, layer, sae_l0, feature)
    
    if is_canonical_sae(sae_width, layer, sae_l0):
        st.components.v1.iframe(dashboard_url_or_path, height=800, scrolling=True)
    else:
        try:
            with open(dashboard_url_or_path, 'r') as file:
                dashboard_html = file.read()

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

            st.components.v1.html(modified_html, height=800, scrolling=True)
        except FileNotFoundError:
            st.error(f"Dashboard for feature {feature} not found. This may be due to the file being missing.")



def plot_sae_probe_cosine_similarities(
    similarities, split_features, absorbing_features
):
    fig = go.Figure()

    # Plot all similarities in light gray
    fig.add_trace(
        go.Scatter(
            y=similarities,
            mode="lines",
            line=dict(color="lightgray"),
            name="Cosine Similarity",
        )
    )

    # Highlight split features in black
    split_x = [i for i in range(len(similarities)) if i in split_features]
    split_y = [similarities[i] for i in split_x]
    fig.add_trace(
        go.Scatter(
            x=split_x,
            y=split_y,
            mode="markers",
            marker=dict(color="black", size=8),
            name="Split Features",
        )
    )

    # Highlight absorbing features in red
    absorption_x = [i for i in range(len(similarities)) if i in absorbing_features]
    absorption_y = [similarities[i] for i in absorption_x]
    fig.add_trace(
        go.Scatter(
            x=absorption_x,
            y=absorption_y,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Absorbing Features",
        )
    )

    fig.update_layout(
        title="SAE Probe Cosine Similarities",
        xaxis_title="Feature Index",
        yaxis_title="Cosine Similarity",
        height=400,
        showlegend=True,
    )
    return fig


def main():
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

    # Count absorbing features for each letter
    absorption_data = load_sae_absorption_data(
        selected_sae_l0, selected_sae_width, selected_layer
    )
    letter_absorbing_features = {}
    for letter in available_letters:
        absorbing_features = absorption_data[(absorption_data["letter"] == letter)][
            "ablation_feat"
        ].nunique()
        letter_absorbing_features[letter] = absorbing_features

    # Create letter options with absorbing feature counts
    letter_options = [
        f"{letter} ({letter_absorbing_features[letter]})"
        for letter in available_letters
    ]

    default_letter = query_params.get("letter", available_letters[0])
    selected_letter_option = st.sidebar.selectbox(
        "Select Letter (count of available absorbing features in parentheses)",
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

    final_df = filtered_df[
        (filtered_df["sae_l0"] == selected_sae_l0)
        & (filtered_df["letter"] == selected_letter)
    ]

    st.subheader(
        f"First Letter Features for Layer {selected_layer}, SAE Width {selected_sae_width}, SAE L0 {selected_sae_l0}"
    )

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

    feature_tokens = (
        letter_absorptions.groupby("ablation_feat")["token"].apply(list).reset_index()
    )

    feature_unique_tokens = {}

    for _, row in feature_tokens.iterrows():
        feature = row["ablation_feat"]
        tokens = row["token"]
        unique_tokens = list(set(tokens))  # Remove duplicates
        feature_unique_tokens[feature] = unique_tokens

    with st.expander("View the raw absorption data"):
        st.write(letter_absorptions)

    sae_probe_cosine_similarities = get_sae_probe_cosine_similarities(
        selected_sae_width, selected_layer, selected_sae_l0, selected_letter
    )

    # Get split features
    split_features = result_df[result_df["letter"] == selected_letter][
        "split_feats"
    ].iloc[0]

    # Get absorbing features
    absorbing_features = letter_absorptions["ablation_feat"].unique()

    st.subheader("Linear Probe & SAE Cosine Similarities")

    # Add a checkbox to toggle the Linear Probe section visibility
    show_linear_probe = st.checkbox("Show Linear Probe Statistics", value=True)

    if show_linear_probe:
        probe_stats = get_probe_stats(selected_layer, selected_letter)
        if not probe_stats.empty:
            precision = probe_stats["precision"].iloc[0]
            recall = probe_stats["recall"].iloc[0]
            f1 = probe_stats["f1"].iloc[0]

            st.write(
                f"Linear probe performance for predicting first letter '{selected_letter}' (ignoring case) at layer {selected_layer}:"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{precision:.3f}")
            col2.metric("Recall", f"{recall:.3f}")
            col3.metric("F1 Score", f"{f1:.3f}")
        else:
            st.write(
                f"No probe statistics available for letter '{selected_letter}' at layer {selected_layer}."
            )

        fig = plot_sae_probe_cosine_similarities(
            sae_probe_cosine_similarities, split_features, absorbing_features
        )
        st.plotly_chart(fig, use_container_width=True)

    selected_letter_feats = result_df[result_df["letter"] == selected_letter][
        "split_feats"
    ].iloc[0]

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader(f"Split features for letter {selected_letter}")

        feats_str = ", ".join([str(feat) for feat in selected_letter_feats])

        feature_str = "feature" if len(selected_letter_feats) == 1 else "features"

        st.write(
            f"The {feature_str} {feats_str} should be the primary 'first letter is {selected_letter}' {feature_str}.",
            f"You should be able to test the activation with random words starting with letter {selected_letter} below.",
            f"\n\nTry finding words that start with {selected_letter} that don't activate the feature.",
            "You can compare them with the tokens we have discovered in the right column.",
        )

    with right_column:
        st.subheader("Absorbing Features")

        if not feature_unique_tokens:
            st.write("No absorbing features found for this selection.")
        else:
            all_unique_tokens = set()
            for tokens in feature_unique_tokens.values():
                all_unique_tokens.update(tokens)

            all_unique_tokens = ",".join(list(all_unique_tokens))

            st.write(
                f"We have discovered that some features capture the 'first letter is {selected_letter}' signal on specific tokens. "
                "Try copying the tokens showing absorption and test their activations on the main feature and compare with the absorbing features."
            )

    left_column_iframe, right_column_iframe = st.columns(2)

    with left_column_iframe:
        feature_tabs = st.tabs(
            [f"Feature: {feature}" for feature in selected_letter_feats]
        )

        for tab, feature in zip(feature_tabs, selected_letter_feats):
            with tab:
                st.write(
                    f"Random '{selected_letter}' tokens from the vocab for testing:"
                )
                st.code(f"{','.join(get_random_letter_tokens(selected_letter))}")

                st.write(
                    f"Random non-{selected_letter} tokens from the vocab for testing:"
                )
                st.code(f"{','.join(get_random_non_letter_tokens(selected_letter))}")

                display_dashboard(
                    selected_sae_width, selected_layer, selected_sae_l0, feature
                )

    with right_column_iframe:
        if len(feature_unique_tokens) > 0:
            feature_tabs = st.tabs(
                [
                    f"Feature: {feature} ({', '.join(tokens)})"
                    for feature, tokens in feature_unique_tokens.items()
                ]
            )

            for tab, (feature, tokens) in zip(
                feature_tabs, feature_unique_tokens.items()
            ):
                with tab:
                    st.write(f"Tokens absorbed by {feature}:")
                    st.code(f"{','.join(tokens)}")

                    st.write("Tokens across all absorbing features:")
                    st.code(all_unique_tokens)

                    display_dashboard(
                        selected_sae_width, selected_layer, selected_sae_l0, feature
                    )
    

if __name__ == "__main__":
    main()
