import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease ML", page_icon="🫀", layout="wide")
st.title("🫀 Prédiction des Maladies Cardiaques")
st.markdown("**IFOAD - Intelligence Artificielle | Dr Arthur Sawadogo**")

@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    return df

@st.cache_resource
def train_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob),
            'model': model
        }
    return results, scaler, X_test, y_test

df = load_data()
results, scaler, X_test, y_test = train_models(df)

page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Performances des modèles",
    "Prédiction patient"
])

if page == "Vue d'ensemble":
    st.subheader("Aperçu du dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total patients", len(df))
    col2.metric("Avec maladie", df['target'].sum())
    col3.metric("Sans maladie", (df['target']==0).sum())
    st.dataframe(df.head(10))

    st.subheader("Distribution de l'âge")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[df['target']==0]['age'], bins=20, alpha=0.7, label='Pas de maladie', color='green')
    ax.hist(df[df['target']==1]['age'], bins=20, alpha=0.7, label='Maladie cardiaque', color='red')
    ax.set_xlabel('Âge')
    ax.set_ylabel('Nombre de patients')
    ax.legend()
    st.pyplot(fig)

elif page == "Performances des modèles":
    st.subheader("Comparaison des algorithmes")
    metrics_df = pd.DataFrame(
        {name: {k: v for k, v in vals.items() if k != 'model'}
         for name, vals in results.items()}
    ).T.round(4)
    st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn'))

    st.subheader("Graphique comparatif")
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c']
    x = np.arange(len(metrics_list))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, vals) in enumerate(results.items()):
        ax.bar(x + i*width, [vals[m] for m in metrics_list],
               width, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metrics_list)
    ax.set_ylim(0, 1.15)
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Prédiction patient":
    st.subheader("Prédire pour un nouveau patient")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Âge", 29, 77, 55)
        sex = st.radio("Sexe", ["Femme (0)", "Homme (1)"])
        sex_val = 1 if "Homme" in sex else 0
        cp = st.selectbox("Type douleur (cp)", [0, 1, 2, 3])
        trestbps = st.slider("Pression artérielle", 94, 200, 130)
    with col2:
        chol = st.slider("Cholestérol", 126, 564, 246)
        fbs = st.radio("Glycémie > 120", ["Non (0)", "Oui (1)"])
        fbs_val = 1 if "Oui" in fbs else 0
        restecg = st.selectbox("ECG au repos", [0, 1, 2])
        thalach = st.slider("Fréq. cardiaque max", 71, 202, 150)
    with col3:
        exang = st.radio("Angine effort", ["Non (0)", "Oui (1)"])
        exang_val = 1 if "Oui" in exang else 0
        oldpeak = st.slider("Dépression ST", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("Pente ST", [0, 1, 2])
        ca = st.selectbox("Nb vaisseaux (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassémie", [1, 2, 3])

    model_choice = st.selectbox("Choisir le modèle", list(results.keys()))

    if st.button("Lancer la prédiction"):
        patient = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                              restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
        patient_scaled = scaler.transform(patient)
        model = results[model_choice]['model']
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]

        if prediction == 1:
            st.error(f"❤️‍🩹 MALADIE CARDIAQUE DÉTECTÉE — Probabilité : {probability[1]:.1%}")
        else:
            st.success(f"💚 PAS DE MALADIE CARDIAQUE — Probabilité : {probability[0]:.1%}")

st.markdown("---")
st.markdown("IFOAD - Intelligence Artificielle | Dr Arthur Sawadogo | 2024")