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

# Plotly specific configuration - equivalent to Seaborn theme for colorblind-friendly palette can be set via color_discrete_sequence

def run_page3():
    st.header("7. Concept of Latent Features in TTS")
    st.markdown("""
    In advanced Text-to-Speech (TTS) systems, raw linguistic features like phonemes, pitch, energy, and duration are often too numerous and complex to be directly used by subsequent synthesis modules. This is where the concept of \"**latent features**\" (also known as embeddings or representations) becomes critical. As described in TTS architectures (e.g., the Encoder section of a TTS paper, page 2), an `Encoder` component transforms these high-dimensional, explicit `Linguistic features` into a lower-dimensional, more abstract representation called `Latent feature`.

    These latent features are not directly interpretable in the same way as raw features (e.g., \"duration in ms\"), but they are incredibly powerful because they capture the complex, underlying relationships and patterns within the speech characteristics. They are crucial for controlling the naturalness and expressiveness of the synthesized audio and serve as a compact, efficient input for the `Decoder` module, which then reconstructs the mel-spectrogram or other acoustic representations.

    This abstraction helps in several ways:
    *   **Dimensionality Reduction**: Reduces the number of variables the model needs to process.
    *   **Feature Compression**: Captures the most salient information in a compact form.
    *   **Complex Relationship Modeling**: Allows the model to learn and represent intricate, non-linear relationships between various acoustic properties.

    For the scope of this application, we will create a *simplified* representation of latent features by standardizing and combining existing numeric data. This will provide a conceptual understanding of how raw features can be transformed into a more abstract, model-friendly format.
    """)

    st.header("8. Simulating Phoneme Feature Embeddings")
    st.markdown("""
    To conceptually demonstrate the creation of \"latent features\" or \"embeddings\" from our raw phoneme characteristics, we will perform a simplified transformation. This process mimics the role of the `Encoder` in a TTS system, where raw, interpretable features are converted into a more abstract, compact representation. This abstraction is essential for machine learning models to effectively learn and process complex speech patterns.

    Our simulation will involve:
    1.  **Standardization**: Scaling our numeric features (duration, pitch, energy) to have a mean of 0 and a standard deviation of 1. This is crucial for many machine learning algorithms, as it prevents features with larger scales from dominating the learning process.
    2.  **Combination**: Creating a new DataFrame where these scaled features serve as our 'latent features'.

    The `StandardScaler` from `sklearn.preprocessing` is ideal for this task, as it transforms data $x$ using the formula:
    $$ z = \frac{x - \mu}{\sigma} $$
    where $\mu$ is the mean of the feature and $\sigma$ is its standard deviation.
    """)

    def create_simple_latent_features(df, numeric_features_list):
        """
        Standardizes specified numeric features from the DataFrame using StandardScaler
        to create 'latent features'.
        """
        features_to_scale = df[numeric_features_list]
        scaler = StandardScaler()
        scaled_features_array = scaler.fit_transform(features_to_scale)
        scaled_df = pd.DataFrame(scaled_features_array, 
                                 columns=numeric_features_list, 
                                 index=df.index)
        return scaled_df

    if st.session_state.get('data_validated', False):
        st.subheader("Generated Latent Features")
        numeric_features = ['duration_ms', 'avg_pitch_hz', 'max_energy']
        st.session_state['latent_features_df'] = create_simple_latent_features(st.session_state['phoneme_df'], numeric_features)
        
        st.write("First 5 rows of the simulated latent features:")
        st.dataframe(st.session_state['latent_features_df'].head())
        st.markdown("""
        The output displays the first 5 rows of our `latent_features_df`. We can observe that the raw features (`duration_ms`, `avg_pitch_hz`, `max_energy`) have been transformed into a standardized scale. Each column now has a mean close to 0 and a standard deviation close to 1.

        This `latent_features_df` conceptually represents a simplified \"embedding\" or \"latent feature\" for each phoneme instance. By standardizing the features, we've removed their original scale and made them comparable, regardless of their initial units or ranges. This transformation is a common preprocessing step in machine learning, and it prepares these features for tasks such as clustering, where distance metrics are sensitive to feature scales.
        """)
    else:
        st.info("Perform data generation and validation first to create latent features.")

    st.header("9. Clustering Phonemes by Similar Characteristics")
    st.markdown("""
    Clustering is an unsupervised machine learning technique that helps us identify natural groupings or segments within our data. In the context of phonetics, clustering phonemes by their acoustic characteristics can be incredibly useful. For language learners, it could help group similar-sounding phonemes that might be easily confused, aiding in targeted pronunciation practice. It also helps us understand inherent phonetic distinctions and similarities within our synthetic dataset.

    We will use the **K-Means algorithm** for clustering. K-Means is an iterative algorithm that partitions $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid). The algorithm aims to minimize the **within-cluster sum of squares (WCSS)**, which is a measure of the variability within each cluster. The objective function is defined as:

    $$ WCSS = \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 $$

    where:
    *   $k$ is the number of clusters.
    *   $S_i$ is the $i$-th cluster.
    *   $x$ is a data point belonging to cluster $S_i$.
    *   $\mu_i$ is the centroid (mean) of cluster $S_i$.

    For this example, we will choose $k=4$ clusters to group our phonemes based on their `latent_features_df` (scaled `duration_ms`, `avg_pitch_hz`, `max_energy`).
    """)

    def perform_kmeans_clustering(data, n_clusters, random_state):
        """
        Applies K-Means clustering to identify groups in data.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        return kmeans.labels_

    if 'latent_features_df' in st.session_state and st.session_state.get('data_validated', False):
        st.subheader("K-Means Clustering")
        num_clusters = st.slider(
            "Number of Clusters ($k$)", min_value=2, max_value=10, value=st.session_state.get('num_clusters', 4), step=1,
            help="Choose the number of clusters for K-Means algorithm. Default is 4."
        )
        if st.button("Perform Clustering", key="perform_clustering_button"):
            st.session_state['num_clusters'] = num_clusters
            with st.spinner(f"Clustering with {num_clusters} clusters..."):
                cluster_labels = perform_kmeans_clustering(
                    st.session_state['latent_features_df'], num_clusters, st.session_state['random_seed']
                )
                st.session_state['phoneme_df']['cluster_label'] = cluster_labels # Add to original df
            st.success("Clustering complete!")
        
        if 'cluster_label' in st.session_state['phoneme_df'].columns:
            st.write("Cluster counts:")
            st.dataframe(st.session_state['phoneme_df']['cluster_label'].value_counts())

            st.write("#### Phoneme Clusters: Duration vs. Pitch")
            fig_cluster = px.scatter(
                st.session_state['phoneme_df'],
                x='duration_ms',
                y='avg_pitch_hz',
                color='cluster_label',
                title='Phoneme Clusters: Duration vs. Pitch',
                labels={'duration_ms': 'Duration (ms)', 'avg_pitch_hz': 'Average Pitch (Hz)'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_cluster.update_traces(marker=dict(size=10, opacity=0.8))
            st.plotly_chart(fig_cluster)

            st.markdown("""
            The clustering results and the scatter plot of 'Duration vs. Pitch' colored by `cluster_label` provide visual insights into how phonemes group based on their characteristics:

            *   **Cluster Counts**: The `value_counts()` output for `cluster_label` shows the distribution of phonemes across the `num_clusters` clusters. This indicates how balanced or unbalanced the groupings are.

            *   **Visual Interpretation**: The scatter plot visually demonstrates the separation of phonemes into distinct clusters. We can observe different regions of the plot corresponding to different cluster labels. For example:
                *   One cluster might primarily contain phonemes with shorter durations and lower pitches (likely consonants).
                *   Another cluster might encompass phonemes with longer durations and higher pitches (likely vowels).
                *   Intermediate clusters could represent phonemes with mixed characteristics or those falling on the boundaries.

            This clustering conceptually aids in categorizing phonemes for educational purposes. For instance, phonemes within the same cluster might be acoustically similar, making them 'difficult to distinguish' phoneme groups for language learners. Identifying such groups can help trainers focus on specific pronunciation challenges and provide targeted exercises.
            """)
    else:
        st.info("Perform data generation, validation, and latent feature creation first to perform clustering.")

    st.header("10. Introducing a Synthetic \"Naturalness Score\"")
    st.markdown("""
    In real-world Text-to-Speech (TTS) systems and language learning applications, the \"naturalness\" or \"quality\" of speech pronunciation is a critical metric. For our synthetic dataset, the `pronunciation_naturalness_score` serves as a proxy for this concept. It's a synthetic metric, ranging from 0 to 100, that quantifies how \"natural\" or \"well-formed\" a phoneme's pronunciation is based on its underlying acoustic characteristics (duration, pitch, energy). This concept is inspired by the discussions around the \"naturalness of voice\" and speech quality in phonetic and TTS research.

    This score will now serve as our **target variable** for a simple predictive modeling task. By building a model to predict this score from the acoustic features, we can conceptually understand how these features contribute to perceived naturalness and how a system might assess or even improve pronunciation quality.

    We will now prepare our data by selecting the relevant features ($X$) and our target variable ($y$), and then split it into training and testing sets to evaluate our model's performance rigorously.
    """)

    if 'phoneme_df' in st.session_state and st.session_state.get('data_validated', False):
        st.subheader("Data Preparation for Modeling")
        X = st.session_state['phoneme_df'][['duration_ms', 'avg_pitch_hz', 'max_energy']]
        y = st.session_state['phoneme_df']['pronunciation_naturalness_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=st.session_state['random_seed'])

        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        st.write(f"Training set size: {len(X_train)} samples")
        st.write(f"Testing set size: {len(X_test)} samples")
        st.success("Features (X) and target (y) prepared and split into training/testing sets.")
        st.markdown("""
        The features ($X$) and the target variable ($y$) have been correctly identified from our `phoneme_df`. The dataset has been successfully split into training and testing sets, with 80% of the data allocated for training the model and 20% for evaluating its performance.

        This preparation ensures that our model will be trained on one subset of the data and then tested on unseen data, providing a more reliable assessment of its generalization capability. We are now ready to train a simple regression model to predict the `pronunciation_naturalness_score`.
        """)
    else:
        st.info("Perform data generation and validation first to prepare data for modeling.")

    st.header("11. Modeling Phoneme Naturalness")
    st.markdown("""
    To understand how phoneme characteristics predict its perceived naturalness, we can employ a regression model. A regression model aims to establish a mathematical relationship between one or more independent variables (our phoneme characteristics) and a dependent variable (the naturalness score). This simulates how a system might quantitatively assess or even provide feedback to improve pronunciation.

    We will use **Linear Regression**, a simple yet powerful and interpretable model. Linear regression models the relationship between a dependent variable $y$ and one or more independent variables $X$ by fitting a linear equation to the observed data. The simple linear regression equation (for a single independent variable) is given by:

    $$ y = \beta_0 + \beta_1 x + \epsilon $$

    And for multiple independent variables (as in our case):

    $$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon $$

    Where:
    *   $y$ is the dependent variable (e.g., `pronunciation_naturalness_score`).
    *   $x_1, x_2, \dots, x_n$ are the independent variables (e.g., `duration_ms`, `avg_pitch_hz`, `max_energy`).
    *   $\beta_0$ is the y-intercept.
    *   $\beta_1, \beta_2, \dots, \beta_n$ are the coefficients (slopes) for each independent variable, indicating the change in $y$ for a one-unit change in $x_i$ while holding other variables constant.
    *   $\epsilon$ is the error term, representing the irreducible error in the model.

    After training, we will evaluate the model's performance using metrics like **R-squared** (proportion of variance explained) and **Mean Absolute Error (MAE)** (average absolute difference between predictions and actual values).
    """)

    def train_simple_regression_model(X_train, y_train):
        """
        Trains a sklearn.linear_model.LinearRegression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_regression_model(model, X_test, y_test):
        """
        Evaluates the performance of a trained regression model.
        """
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return r2, mae

    if all(key in st.session_state for key in ['X_train', 'y_train', 'X_test', 'y_test']):
        st.subheader("Linear Regression Model")
        if st.button("Train and Evaluate Regression Model", key="train_eval_model_button"):
            with st.spinner("Training model..."):
                regression_model = train_simple_regression_model(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['regression_model'] = regression_model
            st.success("Regression Model Training Complete.")

            st.write("### Model Coefficients:")
            for feature, coef in zip(st.session_state['X_train'].columns, regression_model.coef_):
                st.write(f"- {feature}: {coef:.4f}")
            st.write(f"- Intercept: {regression_model.intercept_:.4f}")

            st.write("### Evaluating Model Performance on Test Set:")
            r2, mae = evaluate_regression_model(regression_model, st.session_state['X_test'], st.session_state['y_test'])
            st.write(f"R-squared Score: {r2:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.markdown("""
            The linear regression model has been trained and evaluated:

            *   **Model Coefficients**: The coefficients indicate the synthetic influence of each feature on the `pronunciation_naturalness_score`:
                *   `duration_ms`: A positive coefficient suggests that, all else being equal, longer durations contribute positively to the naturalness score (within the simulated ideal range).
                *   `avg_pitch_hz`: A positive coefficient indicates that higher average pitch also positively influences the naturalness score.
                *   `max_energy`: A relatively larger positive coefficient implies that maximum energy has a strong synthetic positive impact on the naturalness score.
                *   `Intercept`: The baseline naturalness score when all features are zero (though in practice, features won't be zero).

            *   **R-squared Score**: A high R-squared score (e.g., close to 1) indicates that a large proportion of the variance in the `pronunciation_naturalness_score` can be explained by our chosen features. This means the model fits the synthetic data well.

            *   **Mean Absolute Error (MAE)**: The MAE tells us the average magnitude of the errors in a set of predictions, without considering their direction. A low MAE suggests that our model's predictions are, on average, close to the actual naturalness scores.

            These results confirm that, within our synthetic framework, phoneme characteristics like duration, pitch, and energy are strong predictors of pronunciation naturalness. This conceptually demonstrates how a TTS system or a phoneme trainer could assess and predict speech quality based on acoustic features.
            """)
    else:
        st.info("Prepare data for modeling first to train and evaluate the regression model.")

    st.header("12. Interactive Phoneme Analysis")
    st.markdown("""
    Interactive tools are invaluable for language learners and phoneticians alike. They provide a dynamic way to explore and compare phonemes, reinforcing auditory discrimination and phonetic awareness. By allowing users to interact directly with the data, they can gain a deeper, more intuitive understanding of how different acoustic characteristics contribute to the uniqueness of each sound.

    We will create an interactive display that enables users to select a `phoneme_symbol` from a dropdown menu. Upon selection, the application will dynamically display the average `duration_ms`, `avg_pitch_hz`, `max_energy`, and `pronunciation_naturalness_score` for that specific phoneme. Furthermore, it will generate a bar chart comparing these averages against the overall dataset averages, providing immediate context and highlighting distinctive features.
    """)

    if 'phoneme_df' in st.session_state and st.session_state.get('data_validated', False):
        st.subheader("Interactive Phoneme Analyzer")
        unique_phoneme_symbols = sorted(st.session_state['phoneme_df']['phoneme_symbol'].unique().tolist())
        
        selected_phoneme = st.selectbox(
            "Select a Phoneme:", 
            options=unique_phoneme_symbols,
            help="Choose a phoneme symbol to see its average characteristics and comparison."
        )

        if selected_phoneme:
            CHAR_COLS = ['duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
            
            overall_averages = st.session_state['phoneme_df'][CHAR_COLS].mean()
            phoneme_averages = st.session_state['phoneme_df'].groupby('phoneme_symbol')[CHAR_COLS].mean()
            
            selected_phoneme_data = phoneme_averages.loc[selected_phoneme]

            st.markdown(f"### Characteristics for Phoneme: '{selected_phoneme}'")
            for char_col in CHAR_COLS:
                val = selected_phoneme_data.get(char_col)
                if pd.isna(val):
                    st.write(f"**{char_col.replace('_', ' ').title()}:** N/A")
                else:
                    st.write(f"**{char_col.replace('_', ' ').title()}:** {val:.2f}")

            # Generate Bar Chart for comparison using Plotly
            labels = [col.replace('_', ' ').title() for col in CHAR_COLS[:-1]] # Exclude Naturalness Score for bar chart comparison

            phoneme_values_plot = [selected_phoneme_data.get(col, 0) for col in CHAR_COLS[:-1]]
            overall_values_plot = [overall_averages.get(col, 0) for col in CHAR_COLS[:-1]]

            comparison_data = pd.DataFrame({
                'Characteristic': labels * 2,
                'Average Value': phoneme_values_plot + overall_values_plot,
                'Category': [f'Phoneme "{selected_phoneme}"'] * len(labels) + ['Overall Dataset'] * len(labels)
            })
            
            fig_compare = px.bar(comparison_data, x='Characteristic', y='Average Value', color='Category', 
                                 barmode='group', title=f'Comparison of Acoustic Characteristics for Phoneme "{selected_phoneme}" vs. Overall Dataset',
                                 labels={'Characteristic': 'Characteristic', 'Average Value': 'Average Value'},
                                 color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_compare.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_compare)
        
        st.markdown("""
        The interactive display above allows you to dynamically explore the acoustic characteristics of different phonemes in our synthetic dataset. To use it:

        1.  **Select a Phoneme**: Use the dropdown menu to choose any `phoneme_symbol` (e.g., 'a', 's', 'k').
        2.  **Observe Characteristics**: Upon selection, the text output will immediately show the average `duration_ms`, `avg_pitch_hz`, `max_energy`, and `pronunciation_naturalness_score` for that specific phoneme.
        3.  **Compare with Overall Averages**: A bar chart will also be generated, visually comparing the selected phoneme's average acoustic features against the overall averages across the entire dataset. This helps highlight how a particular phoneme stands out (or doesn't) in terms of its characteristics.

        This interactive comparison is a practical tool in a conceptual phoneme trainer. For language learners, it allows for direct exploration of acoustic differences, making it easier to understand why certain sounds might be harder to distinguish or pronounce correctly. For example, you can see if a specific phoneme has a significantly longer duration or higher pitch compared to the average, indicating its unique acoustic signature.
        """)
    else:
        st.info("Perform data generation and validation first to use the interactive analyzer.")

    st.header("13. Conclusion and Key Insights")
    st.markdown("""
    Throughout this application, we've embarked on an interactive journey to explore the fundamental characteristics of phonemes within a synthetic dataset, drawing inspiration from the principles of phonetics and Text-to-Speech (TTS) systems. We've gained several key insights:

    *   **Phoneme Characteristic Distributions**: Our initial visualizations (histograms) revealed the varied distributions of features like `duration_ms`, `avg_pitch_hz`, and `max_energy`, often showing distinct patterns for vowels and consonants. This underscores the diverse acoustic nature of different speech sounds.

    *   **Relationships Between Features**: Scatter plots and pair plots elucidated the interrelationships between these acoustic features. We observed that longer durations often correlate with higher energy, and that categorical factors like `is_vowel` and `dialect` significantly influence pitch, duration, and energy profiles. These correlations are critical for modeling complex speech patterns.

    *   **Categorical Differences**: Bar plots highlighted clear distinctions in average characteristics across `phoneme_symbol` and `dialect`, and between vowels and consonants. This demonstrates how specific sounds and regional variations possess unique acoustic signatures.

    *   **Concept of Latent Features and Clustering**: We simulated the creation of \"latent features\" through standardization, conceptually mirroring how TTS encoders transform raw features into abstract representations. K-Means clustering then allowed us to group phonemes by similar characteristics, illustrating how such techniques could categorize phonemes for targeted language learning.

    *   **Modeling Phoneme Naturalness**: A simple linear regression model successfully predicted a synthetic `pronunciation_naturalness_score` from acoustic features. This demonstrated how data-driven approaches can quantify and assess speech quality, a concept central to improving TTS output and providing feedback in phoneme trainers.

    *   **Interactive Exploration**: The interactive phoneme analyzer showcased the power of Streamlit to create engaging tools for language learners, allowing dynamic comparison of phoneme characteristics against overall averages. This fosters deeper phonetic awareness and supports targeted practice.

    In conclusion, this exploration reinforces how a granular understanding of phoneme characteristics—duration, pitch, and energy—is foundational to both the generation of natural-sounding synthetic speech and the development of effective language learning tools. Data-driven approaches, from visualization to predictive modeling and interactive analysis, offer immense potential for advancing phonetic awareness and speech technology. The insights gained here are directly relevant to the idea of an \"Interactive Phoneme Trainer,\" providing the analytical backbone for such an application.
    """)

    st.header("14. References")
    st.markdown("""
    *   Chowdhury, Md. Jalal Uddin, and Ashab Hussan. \"A review-based study on different Text-to-Speech technologies.\" *International Journal of Computer Science and Network Security* 19.3 (2019): 173-181.
    *   Sadeque, F. Y., Yasar, S., & Islam, M. M. (2013, May). Bangla text to speech conversion: A syllabic unit selection approach. In 2013 International Conference on Informatics, Electronics and Vision (ICIEV) (pp. 1-6). IEEE.
    *   Alam, Firoj, Promila Kanti Nath, & Khan, Mumit (2007). 'Text to speech for Bangla language using festival'. BRAC University.
    *   Alam, Firoj, Promila Kanti Nath, & Khan, Mumit (2011). 'Bangla text to speech using festival', Conference on human language technology for development, pp.154-161.
    *   Arafat, M. Y., Fahrin, S., Islam, M. J., Siddiquee, M. A., Khan, A., Kotwal, M. R. A., & Huda, M. N. (2014, December). Speech synthesis for Bangla text to speech conversion. In The 8th International Conference on Software, Knowledge, Information Management and Applications (SKIMA 2014) (pp. 1-6). IEEE.
    *   Ahmed, K. M., Mandal, P., & Hossain, B. M. (2019). Text to Speech Synthesis for Bangla Language. International Journal of Information Engineering and Electronic Business, 12(2), 1.
    *   Łańcucki, A. (2021). \"Fastpitch: Parallel Text-to-Speech with Pitch Prediction.\" *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 6588-6592. doi: 10.1109/ICASSP39728.2021.9413889.
    *   Luo, R., et al. (2021). \"Lightspeech: Lightweight and Fast Text to Speech with Neural Architecture Search.\" *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 5699-5703. doi: 10.1109/ICASSP39728.2021.9414403.
    *   Alam, Firoj, S. M. Murtoza Habib, & Khan, Mumit (2009). \"Text normalization system for Bangla,\" Proc. of Conf. on Language and Technology, Lahore, pp. 22-24.
    *   Berk, Elliot (2004). JFlex - The Fast Scanner Generator for Java, version 1.4.1. [http://jflex.de](http://jflex.de)
    *   Tran, D., Haines, P., Ma, W., & Sharma, D. (2007, September). Text-to-speech technology-based programming tool. In International Conference On Signal, Speech and Image Processing.
    *   Rashid, M. M., Hussain, M. A., & Rahman, M. S. (2010). Text normalization and diphone preparation for Bangla speech synthesis. Journal of Multimedia, 5(6), 551.
    *   Zeki, M., Khalifa, O. O., & Naji, A. W. (2010, May). Development of an Arabic text-to-speech system. In International Conference on Computer and Communication Engineering (ICCCE'10) (pp. 1-5). IEEE.
    """)
