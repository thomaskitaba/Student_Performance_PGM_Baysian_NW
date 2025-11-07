#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
BN_student_performance.py - Streamlit App

Probabilistic Graphical Modeling of Student Performance using Bayesian Networks
"""

import os
import io
import zipfile
import warnings
import contextlib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# pgmpy: Bayesian network library
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Kaggle downloader helper
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

import streamlit as st

# Suppress noisy warnings for cleaner output
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed discrete features."""
    d = df.copy()

    # 1. Grade gap (G1 - G2)
    d['G1_G2_gap'] = (d['G1'] - d['G2']).clip(lower=-5, upper=5)
    d['G1_G2_gap'] = pd.cut(d['G1_G2_gap'],
                            bins=[-np.inf, -2, 0, 2, np.inf],
                            labels=[0, 1, 2, 3]).astype(int)

    # 2. Average parent education
    d['parent_edu'] = ((d['Medu'] + d['Fedu']) / 2).round().astype(int)

    # 3. Interaction: studytime × parent_edu
    d['study_x_parent'] = d['studytime'] * d['parent_edu']
    d['study_x_parent'] = pd.cut(d['study_x_parent'],
                                 bins=[-1, 2, 5, 9, np.inf],
                                 labels=[0, 1, 2, 3]).astype(int)

    # 4. Binned absences
    d['absences_bin'] = pd.cut(d['absences'],
                               bins=[-1, 1, 5, 15, np.inf],
                               labels=[0, 1, 2, 3]).astype(int)

    # 5. High social activity
    d['goout_high'] = (d['goout'] >= 4).astype(int)

    # 6. Binary target
    d['pass'] = (d['G3'] >= 10).astype(int)

    return d

def download_from_kagglehub(dataset_ref="dskagglemt/student-performance-data-set", 
                           extract_to="student_data"):
    """Download dataset using kagglehub."""
    st.info(f"Downloading dataset from Kaggle: {dataset_ref} ...")

    path = None
    try:
        if hasattr(kagglehub, 'dataset_download'):
            path = kagglehub.dataset_download(dataset_ref)
            st.success(f"Kagglehub returned path: {path}")
        else:
            raise AttributeError("kagglehub.dataset_download not available")
    except Exception as e:
        st.warning(f"Kagglehub unavailable or failed: {e}")
        if os.path.isdir(extract_to):
            csvs = [f for f in os.listdir(extract_to) if f.endswith('.csv')]
            if csvs:
                st.success(f"Found local CSVs in `{extract_to}` — using those files.")
                return extract_to
        raise

    os.makedirs(extract_to, exist_ok=True)

    if os.path.isfile(path) and path.lower().endswith(".zip"):
        st.info("Found ZIP file, extracting...")
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(path=extract_to)
        st.success(f"Extracted ZIP to folder: {extract_to}")
        return extract_to

    if os.path.isdir(path):
        zips = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".zip")]
        if zips:
            zip_path = zips[0]
            st.info(f"Found ZIP inside directory, extracting: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(path=extract_to)
            st.success(f"Extracted ZIP to folder: {extract_to}")
            return extract_to
        else:
            st.info("No ZIP in returned directory — checking for CSV files directly...")
            csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
            if csvs:
                for f in csvs:
                    src = os.path.join(path, f)
                    dst = os.path.join(extract_to, f)
                    if src != dst:
                        with open(src, "rb") as sf, open(dst, "wb") as df:
                            df.write(sf.read())
                st.success(f"Found CSVs and copied to: {extract_to}")
                return extract_to

    raise FileNotFoundError("Could not find or extract dataset from kagglehub output.")

def create_simple_bayesian_network(train_data):
    """Create a simple but robust Bayesian Network"""
    edges = [('G1', 'pass'), ('G2', 'pass'), ('failures', 'pass')]
    model = BayesianNetwork(edges)
    
    # Manually create CPDs with correct shapes
    # CPD for G1 (no parents)
    g1_values = train_data['G1'].value_counts(normalize=True).sort_index()
    g1_cpd = TabularCPD(
        variable='G1',
        variable_card=len(g1_values),
        values=[[p] for p in g1_values.values]
    )
    
    # CPD for G2 (no parents)
    g2_values = train_data['G2'].value_counts(normalize=True).sort_index()
    g2_cpd = TabularCPD(
        variable='G2',
        variable_card=len(g2_values),
        values=[[p] for p in g2_values.values]
    )
    
    # CPD for failures (no parents)
    failures_values = train_data['failures'].value_counts(normalize=True).sort_index()
    failures_cpd = TabularCPD(
        variable='failures',
        variable_card=len(failures_values),
        values=[[p] for p in failures_values.values]
    )
    
    # CPD for pass (with parents G1, G2, failures)
    pass_states = 2
    g1_states = sorted(train_data['G1'].unique())
    g2_states = sorted(train_data['G2'].unique())
    failures_states = sorted(train_data['failures'].unique())
    
    cpt = []
    for g1_val in g1_states:
        for g2_val in g2_states:
            for failure_val in failures_states:
                mask = (train_data['G1'] == g1_val) & (train_data['G2'] == g2_val) & (train_data['failures'] == failure_val)
                subset = train_data[mask]
                
                if len(subset) > 0:
                    pass_probs = subset['pass'].value_counts(normalize=True).sort_index()
                    prob_0 = pass_probs.get(0, 0.0)
                    prob_1 = pass_probs.get(1, 0.0)
                    total = prob_0 + prob_1
                    if total > 0:
                        prob_0 /= total
                        prob_1 /= total
                    else:
                        prob_0, prob_1 = 0.5, 0.5
                else:
                    if g1_val >= 10 and g2_val >= 10 and failure_val == 0:
                        prob_0, prob_1 = 0.1, 0.9
                    else:
                        prob_0, prob_1 = 0.5, 0.5
                
                cpt.append([prob_0, prob_1])
    
    cpt_array = np.array(cpt).T
    pass_cpd = TabularCPD(
        variable='pass',
        variable_card=pass_states,
        values=cpt_array,
        evidence=['G1', 'G2', 'failures'],
        evidence_card=[len(g1_states), len(g2_states), len(failures_states)]
    )
    
    model.add_cpds(g1_cpd, g2_cpd, failures_cpd, pass_cpd)
    return model

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Student Performance Bayesian Network",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎓 Student Performance Prediction using Bayesian Networks")
    st.markdown("""
    This app uses Bayesian Networks to predict student performance based on various factors
    including grades, study habits, and demographic information.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the section", 
                                   ["Data Overview", "Visualizations", "Bayesian Network", "Predictions"])
    
    # ===============================================
    # Data Loading and Processing
    # ===============================================
    @st.cache_data
    def load_data():
        try:
            data_folder = download_from_kagglehub(extract_to="student_data")
            mat_path = os.path.join(data_folder, "student-mat.csv")
            por_path = os.path.join(data_folder, "student-por.csv")
            
            if not os.path.exists(mat_path) or not os.path.exists(por_path):
                st.error("Expected CSV files not found")
                return None
                
            mat = pd.read_csv(mat_path, sep=";")
            por = pd.read_csv(por_path, sep=";")
            data = pd.concat([mat, por], ignore_index=True)
            
            # Encode categorical variables
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].astype('category').cat.codes
            
            # Select relevant features
            keep_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                        'Medu', 'Fedu', 'studytime', 'failures', 'schoolsup',
                        'famsup', 'goout', 'health', 'absences', 'G1', 'G2', 'G3']
            data = data[[c for c in keep_cols if c in data.columns]]
            
            # Apply feature engineering
            data = engineer_features(data)
            data = data.fillna(0).astype('int')
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    data = load_data()
    
    if data is None:
        st.stop()
    
    # ===============================================
    # Data Overview Section
    # ===============================================
    if app_mode == "Data Overview":
        st.header("📊 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Number of students:** {len(data)}")
            st.write(f"**Pass rate:** {data['pass'].mean():.2%}")
            
            st.subheader("Sample Data")
            st.dataframe(data.head(10))
        
        with col2:
            st.subheader("Key Statistics")
            st.dataframe(data[['G1', 'G2', 'G3', 'pass', 'studytime', 'failures', 'absences']].describe())
            
            st.subheader("Pass/Fail Distribution")
            pass_counts = data['pass'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(pass_counts.values, labels=['Fail', 'Pass'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
            ax.set_title('Pass/Fail Distribution')
            st.pyplot(fig)
    
    # ===============================================
    # Visualizations Section
    # ===============================================
    elif app_mode == "Visualizations":
        st.header("📈 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Heatmap")
            key_vars = ['G1', 'G2', 'G3', 'pass', 'studytime', 'failures', 'absences', 'Medu', 'Fedu']
            available_vars = [v for v in key_vars if v in data.columns]
            corr = data[available_vars].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Grade Distributions")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            data['G1'].hist(ax=ax1, bins=20, color='skyblue', alpha=0.7)
            ax1.set_title('G1 Distribution')
            ax1.set_xlabel('G1 Score')
            
            data['G2'].hist(ax=ax2, bins=20, color='lightgreen', alpha=0.7)
            ax2.set_title('G2 Distribution')
            ax2.set_xlabel('G2 Score')
            
            data['G3'].hist(ax=ax3, bins=20, color='lightcoral', alpha=0.7)
            ax3.set_title('G3 Distribution')
            ax3.set_xlabel('G3 Score')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Study Time vs Final Grade")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='studytime', y='G3', data=data, ax=ax)
            ax.set_title('Study Time vs Final Grade (G3)')
            st.pyplot(fig)
    
    # ===============================================
    # Bayesian Network Section
    # ===============================================
    elif app_mode == "Bayesian Network":
        st.header("🕸️ Bayesian Network Analysis")
        
        st.info("Building Bayesian Network... This may take a moment.")
        
        # Use only essential variables
        simple_vars = ['G1', 'G2', 'failures', 'pass']
        simple_data = data[simple_vars].copy()
        
        # Split data
        train_data, test_data = train_test_split(simple_data, test_size=0.3, random_state=42)
        
        try:
            # Create Bayesian network
            bn = create_simple_bayesian_network(train_data)
            is_valid = bn.check_model()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Network Structure")
                st.write("**Edges:**")
                for edge in bn.edges():
                    st.write(f"- {edge[0]} → {edge[1]}")
                
                st.write(f"**Model Validation:** {'✅ PASSED' if is_valid else '⚠️ HAD ISSUES'}")
                
                # Visualize network
                G = nx.DiGraph(bn.edges())
                fig, ax = plt.subplots(figsize=(8, 6))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', 
                       font_size=12, font_weight='bold', arrows=True, ax=ax)
                ax.set_title('Bayesian Network Structure')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Model Performance")
                
                # Make predictions
                infer = VariableElimination(bn)
                test_sample = test_data.head(50)
                y_true = test_sample['pass'].values
                y_pred = []
                
                for _, row in test_sample.iterrows():
                    evidence = {'G1': int(row['G1']), 'G2': int(row['G2']), 'failures': int(row['failures'])}
                    try:
                        res = infer.map_query(variables=['pass'], evidence=evidence, show_progress=False)
                        y_pred.append(res['pass'])
                    except Exception:
                        y_pred.append(train_data['pass'].mode()[0])
                
                if len(y_pred) > 0:
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        'Value': [acc, prec, rec, f1]
                    })
                    
                    st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}))
                    
                    # Confusion matrix
                    fig, ax = plt.subplots(figsize=(6, 4))
                    cm = confusion_matrix(y_true, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
                    disp.plot(ax=ax, cmap='Blues')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error building Bayesian Network: {e}")
    
    # ===============================================
    # Predictions Section
    # ===============================================
    elif app_mode == "Predictions":
        st.header("🔮 Make Predictions")
        
        st.info("Using the trained Bayesian Network to predict student performance")
        
        # Build model for predictions
        simple_vars = ['G1', 'G2', 'failures', 'pass']
        simple_data = data[simple_vars].copy()
        train_data, _ = train_test_split(simple_data, test_size=0.3, random_state=42)
        
        try:
            bn = create_simple_bayesian_network(train_data)
            infer = VariableElimination(bn)
            
            st.subheader("Enter Student Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                g1 = st.slider("First Period Grade (G1)", 0, 20, 10)
                g2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
            
            with col2:
                failures = st.slider("Number of Past Failures", 0, 4, 0)
                studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], 
                                       format_func=lambda x: f"{x} - {['<2h', '2-5h', '5-10h', '>10h'][x-1]}")
            
            with col3:
                absences = st.slider("Number of Absences", 0, 30, 5)
                parent_edu = st.selectbox("Parent Education Level", [0, 1, 2, 3, 4],
                                        format_func=lambda x: f"Level {x}")
            
            if st.button("Predict Student Performance"):
                evidence = {
                    'G1': g1,
                    'G2': g2,
                    'failures': failures
                }
                
                try:
                    result = infer.map_query(variables=['pass'], evidence=evidence, show_progress=False)
                    prediction = result['pass']
                    
                    # Get probability distribution
                    prob_dist = infer.query(variables=['pass'], evidence=evidence)
                    
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.success(f"🎯 Prediction: **PASS** (Student is likely to pass)")
                        else:
                            st.error(f"🎯 Prediction: **FAIL** (Student is at risk of failing)")
                    
                    with col2:
                        st.metric("Probability of Passing", f"{prob_dist.values[1]:.2%}")
                        st.metric("Probability of Failing", f"{prob_dist.values[0]:.2%}")
                    
                    # Additional insights
                    st.subheader("Insights & Recommendations")
                    if prediction == 0:
                        st.warning("""
                        **Recommendations for at-risk student:**
                        - Consider additional tutoring or academic support
                        - Monitor attendance and study habits
                        - Schedule regular progress reviews
                        - Provide targeted interventions
                        """)
                    else:
                        st.success("""
                        **Student is on track:**
                        - Continue current study habits
                        - Maintain good attendance
                        - Consider advanced coursework opportunities
                        """)
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        
        except Exception as e:
            st.error(f"Error setting up prediction model: {e}")

if __name__ == "__main__":
    main()