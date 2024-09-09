import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")


@st.cache_data
def load_available_sae_l0s():
    return pd.read_parquet("data/sae_split_feats.parquet")


@st.cache_data
def load_full_data():
    return pd.read_parquet("data/feature_absorption_results.parquet")

@st.cache_data
def load_sae_data(sae_l0, sae_width, layer, letter):
    df = load_full_data()
    return df[
        (df["sae_l0"] == sae_l0)
        & (df["sae_width"] == sae_width)
        & (df["layer"] == layer)
        & (df["letter"] == letter)
    ]


layer_l0_dict = {
    16000: {0: 105, 1: 102, 2: 141, 3: 59, 4: 124, 5: 68, 6: 70, 7: 69, 8: 71, 9: 73},
    65000: {0: 73, 1: 121, 2: 77, 3: 89, 4: 89, 5: 105, 6: 107, 7: 107, 8: 111, 9: 118},
}


def main():
    st.title("Feature Absorption Results Explorer")

    available_saes_df = load_available_sae_l0s()

    # Move selectors to the sidebar
    st.sidebar.subheader("Select an SAE and the first letter to explore")

    layers = sorted(available_saes_df["layer"].unique())
    selected_layer = st.sidebar.selectbox("Select Layer", layers, key="layer")

    sae_widths = sorted(available_saes_df["sae_width"].unique())
    selected_sae_width = st.sidebar.selectbox(
        "Select SAE Width", sae_widths, key="sae_width"
    )

    filtered_df = available_saes_df[
        (available_saes_df["layer"] == selected_layer)
        & (available_saes_df["sae_width"] == selected_sae_width)
    ]

    selected_sae_l0 = layer_l0_dict[selected_sae_width][selected_layer]

    available_letters = filtered_df[filtered_df["sae_l0"] == selected_sae_l0][
        "letter"
    ].unique()

    # Check if the previously selected letter is still available
    if (
        "selected_letter" in st.session_state
        and st.session_state.selected_letter in available_letters
    ):
        default_letter_index = list(available_letters).index(
            st.session_state.selected_letter
        )
    else:
        default_letter_index = 0

    selected_letter = st.sidebar.selectbox(
        "Select Letter", available_letters, index=default_letter_index, key="letter"
    )

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

    sae_data = load_sae_data(
        selected_sae_l0, selected_sae_width, selected_layer, selected_letter
    )

    sae_data_only_absorptions = sae_data[
        (sae_data["feat_order"] == 0) & (sae_data["is_absorption"])
    ]

    feature_tokens = (
        sae_data_only_absorptions.groupby("ablation_feat")["token"]
        .apply(list)
        .reset_index()
    )

    feature_unique_tokens = {}

    for _, row in feature_tokens.iterrows():
        feature = row["ablation_feat"]
        tokens = row["token"]
        unique_tokens = list(set(tokens))  # Remove duplicates
        feature_unique_tokens[feature] = unique_tokens

    with st.expander("View the raw absorption data"):
        st.write(sae_data_only_absorptions)

    sae_link_part = f"{selected_layer}-gemmascope-res-{selected_sae_width // 1000}k"

    selected_letter_feats = result_df[result_df["letter"] == selected_letter][
        "split_feats"
    ].iloc[0]

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader(f"Split features for letter {selected_letter}:")

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

            st.write("All tokens showing absorption (for copying):")
            st.code(all_unique_tokens)

    left_column_iframe, right_column_iframe = st.columns(2)

    with left_column_iframe:
        feature_tabs = st.tabs(
            [f"Feature: {feature}" for feature in selected_letter_feats]
        )

        for tab, feature in zip(feature_tabs, selected_letter_feats):
            with tab:
                st.write("Tokens:")
                st.code(f"Should activate on most '{selected_letter}' tokens")
                iframe_url = f"https://neuronpedia.org/gemma-2-2b/{sae_link_part}/{feature}?embed=true"
                # Display the iframe using custom HTML with st.components.v1.html()
                st.components.v1.html(
                    f"""
                    <div style="position: relative; height: 100vh; overflow: hidden;">
                        <iframe src="{iframe_url}" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                        frameborder="0" 
                        scrolling="yes">
                           </iframe>
                    </div>
        """,
                    height=800,  # This sets a default height, but the iframe will expand to fill the viewport
                    scrolling=True,
                )

    with right_column_iframe:
        # Create tabs for absorbing features
        feature_tabs = st.tabs(
            [
                f"Feature: {feature} ({', '.join(tokens)})"
                for feature, tokens in feature_unique_tokens.items()
            ]
        )

        for tab, (feature, tokens) in zip(feature_tabs, feature_unique_tokens.items()):
            with tab:
                st.write("Tokens:")
                st.code(f"{','.join(tokens)}")
                iframe_url = f"https://neuronpedia.org/gemma-2-2b/{sae_link_part}/{feature}?embed=true"
                st.components.v1.html(
                    f"""
                    <div style="position: relative; height: 100vh; overflow: hidden;">
                        <iframe src="{iframe_url}" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                        frameborder="0" 
                        scrolling="yes">
                        </iframe>
                    </div>
                    """,
                    height=800,  # This sets a default height, but the iframe will expand to fill the viewport
                    scrolling=True,
                )


# ... rest of the code ...

if __name__ == "__main__":
    main()
