import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error 
import io

# Configure Plotly aesthetics (equivalent to Seaborn theme for colorblind-friendly palette)
# Note: Plotly themes are applied differently. We use color_discrete_sequence for consistency.

def run_page2():
    st.header("4. Visualizing Phoneme Durations and Frequencies")
    st.markdown("""
    Visualizing the distribution of phoneme characteristics is a fundamental step in phonetic analysis. It helps us understand the typical ranges, variability, and overall patterns within features like duration, pitch, and energy. For instance, `duration_ms` is a key feature influencing the rhythm and clarity of speech, as noted in discussions of \"Phoneme duration\" in TTS research.

    We will generate histograms for `duration_ms`, `avg_pitch_hz`, and `max_energy` to observe their individual distributions.
    """)

    def plot_distribution_histogram(df, column_name, title, x_label, y_label, bins=30):
        fig = px.histogram(df, x=column_name, nbins=bins, title=title, 
                           labels={column_name: x_label, "count": y_label}, 
                           color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(bargap=0.1) # Add some gap between bars for better visualization
        return fig

    if st.session_state.get('data_validated', False):
        st.subheader("Distribution Histograms")
        bins_input = st.slider("Number of Bins for Histograms", min_value=10, max_value=50, value=30, step=5,
                               help="Adjust the number of bins to see different levels of detail in the distributions.")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig1 = plot_distribution_histogram(
                st.session_state['phoneme_df'], 'duration_ms', 'Distribution of Phoneme Duration (ms)', 'Duration (ms)', 'Frequency', bins=bins_input
            )
            st.plotly_chart(fig1)
        with col2:
            fig2 = plot_distribution_histogram(
                st.session_state['phoneme_df'], 'avg_pitch_hz', 'Distribution of Average Pitch (Hz)', 'Average Pitch (Hz)', 'Frequency', bins=bins_input
            )
            st.plotly_chart(fig2)
        with col3:
            fig3 = plot_distribution_histogram(
                st.session_state['phoneme_df'], 'max_energy', 'Distribution of Maximum Energy', 'Maximum Energy', 'Frequency', bins=bins_input
            )
            st.plotly_chart(fig3)
        st.markdown("""
        The generated histograms provide insights into the synthetic distributions of key phoneme characteristics:

        *   **Duration ($\text{ms}$)**: The histogram for `duration_ms` appears to be a bimodal or multimodal distribution, reflecting the synthetic distinction between generally shorter consonants and longer vowels. The peaks align with the mean durations set for vowels (around 120ms) and consonants (around 60ms).
        *   **Average Pitch ($\text{Hz}$)**: The `avg_pitch_hz` distribution also shows distinct patterns, likely influenced by the synthetic assignment of different pitch ranges to vowels and consonants, or specific phonemes. This aligns with the understanding that pitch varies significantly across different speech sounds.
        *   **Maximum Energy**: The distribution of `max_energy` also indicates clear differences between phoneme types, with vowels typically exhibiting higher energy. The histogram shows a spread that covers both lower-energy consonants and higher-energy vowels.

        These distributions confirm that our synthetic data successfully introduces realistic variability and categorical distinctions, which is crucial for simulating phonetic phenomena for a phoneme trainer or TTS system. The varied shapes imply that these features are not uniformly distributed and are influenced by underlying factors, such as the `is_vowel` attribute and specific `phoneme_symbol` adjustments.
        """)
    else:
        st.info("Perform data generation and validation first to visualize distributions.")

    st.header("5. Analyzing Relationships: Pitch, Energy, and Duration")
    st.markdown("""
    Understanding the interrelationships between different speech features is vital for comprehending how phonemes are formed and perceived. For example, research highlights that \"Pitch: Key feature to convey emotions, it greatly affects the speech prosody\" and \"Energy: Indicates frame-level magnitude of mel-spectrograms... affects the volume and prosody of speech.\" Exploring these correlations helps us build predictive models, identify patterns in phoneme pronunciation, and understand the acoustic cues that differentiate sounds.

    We will use scatter plots to visualize pairwise relationships, optionally coloring points by categorical variables like `is_vowel` or `dialect` to observe how these categories influence the relationships. Additionally, a `plotly.express.scatter_matrix` will provide a comprehensive overview of all pairwise relationships and distributions for multiple numeric features.
    """)

    def plot_scatter_relationship(df, x_column, y_column, hue_column, title, x_label, y_label):
        fig = px.scatter(df, x=x_column, y=y_column, color=hue_column, title=title, 
                         labels={x_column: x_label, y_column: y_label}, 
                         color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        return fig

    def plot_features_pairplot(df, features_list, hue_column=None, title=None):
        if df.empty or not features_list:
            st.warning("Cannot plot an empty DataFrame or with no specified features.")
            return None
        
        fig = px.scatter_matrix(df, dimensions=features_list, color=hue_column, title=title,
                                color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(title_y=0.99, title_x=0.5, font_size=12)
        return fig

    if st.session_state.get('data_validated', False):
        st.subheader("Relationship Plots")

        st.write("#### Duration vs. Pitch, by Vowel/Consonant")
        fig_s1 = plot_scatter_relationship(
            st.session_state['phoneme_df'], 'duration_ms', 'avg_pitch_hz', 'is_vowel', 
            'Duration vs. Pitch, by Vowel/Consonant', 'Duration (ms)', 'Average Pitch (Hz)'
        )
        st.plotly_chart(fig_s1)

        st.write("#### Max Energy vs. Pitch, by Dialect")
        fig_s2 = plot_scatter_relationship(
            st.session_state['phoneme_df'], 'max_energy', 'avg_pitch_hz', 'dialect', 
            'Max Energy vs. Pitch, by Dialect', 'Maximum Energy', 'Average Pitch (Hz)'
        )
        st.plotly_chart(fig_s2)

        st.subheader("Pair Plot of Phoneme Characteristics")
        numeric_cols_for_pairplot = ['duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
        selected_pairplot_features = st.multiselect(
            "Select Features for Pair Plot", 
            options=numeric_cols_for_pairplot, 
            default=st.session_state.get('selected_pairplot_features', numeric_cols_for_pairplot),
            help="Choose the numeric features to display in the pair plot."
        )
        pairplot_hue_options = ['None'] + [col for col in st.session_state['phoneme_df'].select_dtypes(include='object').columns if col != 'phoneme_symbol'] + ['is_vowel']
        selected_pairplot_hue = st.selectbox(
            "Color Pair Plot by (Hue)", 
            options=pairplot_hue_options, 
            index=pairplot_hue_options.index(st.session_state.get('selected_pairplot_hue', 'is_vowel')),
            help="Select a categorical column to color the points in the pair plot."
        )
        pairplot_hue_val = None if selected_pairplot_hue == 'None' else selected_pairplot_hue

        if st.button("Generate Pair Plot", key="pairplot_button"):
            st.session_state['selected_pairplot_features'] = selected_pairplot_features
            st.session_state['selected_pairplot_hue'] = selected_pairplot_hue
            with st.spinner("Generating pair plot... This might take a moment."):
                fig_pair = plot_features_pairplot(
                    st.session_state['phoneme_df'],
                    selected_pairplot_features,
                    hue_column=pairplot_hue_val,
                    title='Pair Plot of Phoneme Characteristics'
                )
                if fig_pair:
                    st.plotly_chart(fig_pair)
                else:
                    st.warning("Could not generate pair plot. Check selected features/data.")

        st.markdown("""
        The scatter plots and the pair plot reveal interesting synthetic relationships within our phoneme data:

        *   **Duration vs. Pitch (by Vowel/Consonant)**: The scatter plot clearly shows a distinction between vowels and consonants. Vowels (True for `is_vowel`) tend to occupy the upper-right region of the plot, indicating generally longer durations and higher average pitches. Consonants (False for `is_vowel`) are typically in the lower-left, with shorter durations and lower pitches. This aligns with phonetic principles where vowels are often sustained longer and have a clearer fundamental frequency.

        *   **Max Energy vs. Pitch (by Dialect)**: This plot highlights how different synthetic `dialect` categories are distributed across energy and pitch ranges. While there's significant overlap, subtle clustering or shifts for certain dialects might be observed, reflecting simulated regional variations in speech production. For instance, one dialect might be characterized by slightly higher overall pitch or energy compared to another.

        *   **Pair Plot of Phoneme Characteristics**: The `plotly.express.scatter_matrix` offers a holistic view:
            *   **Histograms on the diagonal** reaffirm the distributions seen earlier, often showing bimodal patterns for features like `duration_ms` and `avg_pitch_hz` when colored by `is_vowel`.
            *   **Scatter plots off-diagonal** further illustrate pairwise correlations. For example, there appears to be a positive correlation between `duration_ms` and `max_energy` (longer phonemes tend to be more energetic). The `pronunciation_naturalness_score` also shows positive correlations with these acoustic features, suggesting that 'naturalness' is synthetically linked to optimal ranges of duration, pitch, and energy.

        These visualizations are critical for understanding how different phonetic features interact and how categorical factors like vowel/consonant type or dialect influence these interactions. Such insights are fundamental for developing robust TTS systems or effective phoneme training tools.
        """)
    else:
        st.info("Perform data generation and validation first to analyze relationships.")

    st.header("6. Categorical Comparisons: Phoneme Type and Dialect")
    st.markdown("""
    Comparing phoneme characteristics across different categories, such as `phoneme_symbol`, `is_vowel`, or `dialect`, is essential for identifying distinct patterns and variations in speech sounds. This analysis helps us understand how individual phonemes behave, how vowels differ from consonants, and how regional accents might manifest acoustically. Such insights are crucial for developing a robust phoneme trainer or a highly accurate TTS system that can account for phonetic distinctions and regional variations.
    """)

    def plot_categorical_bar_comparison(df, category_column, value_column, title, x_label, y_label):
        grouped_data = df.groupby(category_column)[value_column].mean().reset_index()
        fig = px.bar(grouped_data, x=category_column, y=value_column, title=title, 
                     labels={category_column: x_label, value_column: y_label},
                     color=category_column, color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(xaxis_tickangle=-45) # Rotate x-axis labels if many categories
        return fig

    if st.session_state.get('data_validated', False):
        st.subheader("Bar Charts for Categorical Comparisons")
        
        st.write("#### Average Phoneme Duration by Symbol")
        fig_b1 = plot_categorical_bar_comparison(
            st.session_state['phoneme_df'], 'phoneme_symbol', 'duration_ms', 
            'Average Phoneme Duration by Symbol', 'Phoneme Symbol', 'Average Duration (ms)'
        )
        st.plotly_chart(fig_b1)

        st.write("#### Average Pitch by Dialect")
        fig_b2 = plot_categorical_bar_comparison(
            st.session_state['phoneme_df'], 'dialect', 'avg_pitch_hz', 
            'Average Pitch by Dialect', 'Dialect', 'Average Pitch (Hz)'
        )
        st.plotly_chart(fig_b2)

        st.write("#### Average Naturalness Score: Vowels vs. Consonants")
        fig_b3 = plot_categorical_bar_comparison(
            st.session_state['phoneme_df'], 'is_vowel', 'pronunciation_naturalness_score', 
            'Average Naturalness Score: Vowels vs. Consonants', 'Is Vowel?', 'Average Naturalness Score'
        )
        st.plotly_chart(fig_b3)

        st.markdown("""
        The bar plots provide clear comparisons across different categorical groups:

        *   **Average Phoneme Duration by Symbol**: This plot shows how the average `duration_ms` varies for each `phoneme_symbol`. We can observe that vowels (e.g., 'a', 'e', 'i', 'o', 'u') generally have longer average durations compared to consonants. This reflects the synthetic generation logic where vowels were designed to be more sustained.

        *   **Average Pitch by Dialect**: The plot comparing `avg_pitch_hz` across different `dialect` categories reveals subtle (synthetic) variations. For instance, 'General American' might have a slightly different average pitch compared to 'British English' or 'Australian English'. These differences, while synthetic, illustrate how regional variations can impact fundamental speech characteristics.

        *   **Average Naturalness Score: Vowels vs. Consonants**: This bar plot effectively shows whether `is_vowel` (True/False) correlates with the `pronunciation_naturalness_score`. Given our synthetic generation, vowels likely have a higher average naturalness score, as they are often more acoustically prominent and central to prosody. This comparison highlights how different types of phonemes might inherently contribute differently to perceived speech quality.

        These observations are critical for understanding how specific phoneme types and dialects contribute to the overall acoustic properties of speech. Such insights are directly applicable to a phoneme trainer, where learners could focus on specific phoneme categories or dialectal pronunciations based on their distinct characteristics.
        """)
    else:
        st.info("Perform data generation and validation first to visualize categorical comparisons.")