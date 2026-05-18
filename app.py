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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease AI", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background: #0f1117; color: #e8e8e8; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #1e2740; }
.hero { background: linear-gradient(135deg, #1a1f35 0%, #0f1117 50%, #1a1228 100%); border: 1px solid #1e2740; border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden; }
.hero::before { content: ''; position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: radial-gradient(circle, rgba(229,57,53,0.15) 0%, transparent 70%); border-radius: 50%; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #ffffff; margin: 0; line-height: 1.1; }
.hero-title span { color: #ef5350; }
.hero-sub { color: #6b7594; font-size: 0.95rem; margin-top: 0.5rem; font-weight: 300; letter-spacing: 0.05em; }
.hero-badge { display: inline-block; background: rgba(239,83,80,0.15); border: 1px solid rgba(239,83,80,0.3); color: #ef8a87; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 500; margin-bottom: 1rem; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card { background: #161b27; border: 1px solid #1e2740; border-radius: 16px; padding: 1.4rem; text-align: center; }
.metric-num { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #ffffff; line-height: 1; }
.metric-label { color: #6b7594; font-size: 12px; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-accent { color: #ef5350; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: #ffffff; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1e2740; }
.result-danger { background: linear-gradient(135deg, #2d1515, #1a0f0f); border: 1px solid #ef5350; border-radius: 16px; padding: 2rem; text-align: center; }
.result-safe { background: linear-gradient(135deg, #0f2d1a, #0a1f12); border: 1px solid #26a69a; border-radius: 16px; padding: 2rem; text-align: center; }
.rank-row { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; background: #161b27; border: 1px solid #1e2740; }
.rank-num { font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: #3a4160; width: 28px; text-align: center; }
.rank-name { flex: 1; color: #c8cde0; font-size: 14px; font-weight: 500; }
.rank-score { background: rgba(239,83,80,0.15); color: #ef8a87; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.nav-label { color: #6b7594; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; padding: 0 0.5rem; }
.stButton > button { background: linear-gradient(135deg, #ef5350, #c62828) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 500 !important; padding: 0.6rem 2rem !important; width: 100% !important; font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#161b27', 'axes.facecolor': '#161b27',
    'axes.edgecolor': '#1e2740', 'axes.labelcolor': '#a0a8c0',
    'text.color': '#c8cde0', 'xtick.color': '#6b7594', 'ytick.color': '#6b7594',
    'grid.color': '#1e2740', 'axes.grid': True,
    'axes.spines.top': False, 'axes.spines.right': False,
})
COLORS = ['#ef5350', '#26a69a', '#7e57c2', '#ffa726', '#42a5f5', '#66bb6a']

@st.cache_data
def load_data():
    try:
        return pd.read_csv('heart.csv')
    except:
        np.random.seed(42); n = 297
        age = np.random.randint(29,77,n); sex = np.random.choice([0,1],n,p=[0.32,0.68])
        cp = np.random.choice([0,1,2,3],n,p=[0.47,0.17,0.28,0.08])
        trestbps = np.random.normal(131,17,n).clip(94,200).astype(int)
        chol = np.random.normal(246,51,n).clip(126,564).astype(int)
        fbs = np.random.choice([0,1],n,p=[0.85,0.15]); restecg = np.random.choice([0,1,2],n,p=[0.48,0.49,0.03])
        thalach = np.random.normal(150,22,n).clip(71,202).astype(int)
        exang = np.random.choice([0,1],n,p=[0.67,0.33])
        oldpeak = np.round(np.random.exponential(1.0,n).clip(0,6.2),1)
        slope = np.random.choice([0,1,2],n,p=[0.21,0.46,0.33])
        ca = np.random.choice([0,1,2,3],n,p=[0.58,0.22,0.13,0.07])
        thal = np.random.choice([1,2,3],n,p=[0.06,0.54,0.40])
        score = (-0.02*(age-50)+0.3*(cp==0)-0.4*(cp==3)+0.3*exang-0.01*(thalach-150)+0.3*oldpeak+0.2*(ca>0)+np.random.normal(0,0.5,n))
        target = (score > 0.1).astype(int)
        return pd.DataFrame({'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal,'target':target})

@st.cache_resource
def train_models(df):
    X = df.drop('target',axis=1); y = df['target']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf',probability=True,random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42,max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100,random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100,random_state=42)
    }
    results = {}
    for name,model in models.items():
        model.fit(X_train_s,y_train)
        y_pred = model.predict(X_test_s); y_prob = model.predict_proba(X_test_s)[:,1]
        results[name] = {'Accuracy':accuracy_score(y_test,y_pred),'Precision':precision_score(y_test,y_pred),
            'Recall':recall_score(y_test,y_pred),'F1-Score':f1_score(y_test,y_pred),
            'AUC-ROC':roc_auc_score(y_test,y_prob),'model':model,'y_pred':y_pred,'y_prob':y_prob}
    return results, scaler, X_test, y_test

df = load_data()
results, scaler, X_test, y_test = train_models(df)

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#fff;">🫀 HeartAI</div>
        <div style="color:#6b7594;font-size:12px;margin-top:4px;">Machine Learning Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠  Tableau de bord","🔍  Analyse des données","📊  Performances","🩺  Prédiction"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""<div style="color:#3a4160;font-size:11px;line-height:1.8;">
        <div>Dataset : Heart Disease UCI</div><div>Patients : {len(df)}</div>
        <div>Algorithmes : 6</div><div>Meilleur AUC : {max(v['AUC-ROC'] for v in results.values()):.4f}</div>
    </div>""", unsafe_allow_html=True)

if "Tableau" in page:
    st.markdown("""<div class="hero">
        <div class="hero-badge">Intelligence Artificielle · Classification</div>
        <div class="hero-title">Prédiction des<br><span>Maladies Cardiaques</span></div>
        <div class="hero-sub">Heart Disease UCI Dataset · 6 algorithmes de Machine Learning</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="metric-grid">
        <div class="metric-card"><div class="metric-num">{len(df)}</div><div class="metric-label">Patients</div></div>
        <div class="metric-card"><div class="metric-num">14</div><div class="metric-label">Variables</div></div>
        <div class="metric-card"><div class="metric-num metric-accent">{df['target'].sum()}</div><div class="metric-label">Cas positifs</div></div>
        <div class="metric-card"><div class="metric-num">6</div><div class="metric-label">Modèles</div></div>
    </div>""", unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-title">Aperçu du dataset</div>', unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True, height=280)
    with col2:
        st.markdown('<div class="section-title">Distribution cible</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#161b27'); ax.set_facecolor('#161b27')
        wedges, texts, autotexts = ax.pie([df['target'].sum(),(df['target']==0).sum()],
            labels=['Maladie','Sain'], autopct='%1.1f%%', colors=['#ef5350','#26a69a'],
            wedgeprops={'linewidth':3,'edgecolor':'#0f1117'}, startangle=90,
            textprops={'color':'#c8cde0','fontsize':13})
        for at in autotexts: at.set_fontsize(12); at.set_fontweight('bold')
        st.pyplot(fig); plt.close()
    st.markdown('<div class="section-title">Classement des modèles</div>', unsafe_allow_html=True)
    ranking = sorted(results.items(), key=lambda x: x[1]['F1-Score'], reverse=True)
    medals = ['🥇','🥈','🥉','4.','5.','6.']
    cols = st.columns(2)
    for i,(name,vals) in enumerate(ranking):
        with cols[i%2]:
            st.markdown(f"""<div class="rank-row">
                <div class="rank-num">{medals[i]}</div>
                <div class="rank-name">{name}</div>
                <div class="rank-score">F1 {vals['F1-Score']:.3f}</div>
                <div style="color:#3a4160;font-size:12px;margin-left:6px;">AUC {vals['AUC-ROC']:.3f}</div>
            </div>""", unsafe_allow_html=True)

elif "Analyse" in page:
    st.markdown('<div class="hero-title" style="font-family:\'DM Serif Display\',serif;color:#fff;font-size:2rem;margin-bottom:1.5rem;">Analyse <span style="color:#ef5350;">Exploratoire</span></div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    fig.patch.set_facecolor('#0f1117')
    def plot_hist(ax, col, title):
        ax.hist(df[df['target']==0][col],bins=18,alpha=0.8,label='Sain',color='#26a69a')
        ax.hist(df[df['target']==1][col],bins=18,alpha=0.8,label='Maladie',color='#ef5350')
        ax.set_title(title,color='#c8cde0',fontsize=11,fontweight='bold',pad=10)
        ax.legend(fontsize=9,framealpha=0); ax.set_facecolor('#161b27')
        ax.spines['bottom'].set_color('#1e2740'); ax.spines['left'].set_color('#1e2740')
    plot_hist(axes[0,0],'age','Distribution de l\'âge')
    plot_hist(axes[0,1],'thalach','Fréquence cardiaque max')
    plot_hist(axes[0,2],'chol','Cholestérol')
    plot_hist(axes[1,0],'trestbps','Pression artérielle')
    plot_hist(axes[1,1],'oldpeak','Dépression ST')
    sex_d = df.groupby(['sex','target']).size().unstack(fill_value=0)
    sex_d.index=['Femme','Homme']; sex_d.columns=['Sain','Maladie']
    sex_d.plot(kind='bar',ax=axes[1,2],color=['#26a69a','#ef5350'],alpha=0.85,rot=0)
    axes[1,2].set_title('Maladie par sexe',color='#c8cde0',fontsize=11,fontweight='bold',pad=10)
    axes[1,2].set_facecolor('#161b27'); axes[1,2].legend(fontsize=9,framealpha=0)
    axes[1,2].spines['bottom'].set_color('#1e2740'); axes[1,2].spines['left'].set_color('#1e2740')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="section-title">Matrice de corrélation</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12,8))
    fig2.patch.set_facecolor('#0f1117'); ax2.set_facecolor('#161b27')
    mask = np.triu(np.ones_like(df.corr(),dtype=bool))
    sns.heatmap(df.corr(),mask=mask,annot=True,fmt='.2f',cmap='RdYlGn',center=0,vmin=-1,vmax=1,
                linewidths=0.5,linecolor='#0f1117',square=True,ax=ax2,
                annot_kws={'size':9,'color':'#c8cde0'},cbar_kws={'shrink':0.8})
    ax2.tick_params(colors='#6b7594'); plt.tight_layout(); st.pyplot(fig2); plt.close()

elif "Performances" in page:
    st.markdown('<div class="hero-title" style="font-family:\'DM Serif Display\',serif;color:#fff;font-size:2rem;margin-bottom:1.5rem;">Performances <span style="color:#ef5350;">des Modèles</span></div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({name:{k:v for k,v in vals.items() if k not in ['model','y_pred','y_prob']} for name,vals in results.items()}).T.astype(float).round(4)
    st.dataframe(metrics_df[['Accuracy','Precision','Recall','F1-Score','AUC-ROC']].style.background_gradient(cmap='RdYlGn',subset=['Accuracy','Precision','Recall','F1-Score','AUC-ROC']).format('{:.4f}'),use_container_width=True,height=250)
    tab1,tab2,tab3 = st.tabs(["📊 Comparaison","📈 Courbes ROC","🔲 Confusion"])
    with tab1:
        metrics_list=['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        x=np.arange(len(metrics_list)); width=0.13
        fig,ax=plt.subplots(figsize=(14,6)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#161b27')
        for i,(name,vals) in enumerate(results.items()):
            ax.bar(x+i*width,[vals[m] for m in metrics_list],width,label=name,color=COLORS[i],alpha=0.9)
        ax.set_xticks(x+width*2.5); ax.set_xticklabels(metrics_list,fontsize=11); ax.set_ylim(0,1.15)
        ax.spines['bottom'].set_color('#1e2740'); ax.spines['left'].set_color('#1e2740')
        ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=9,framealpha=0,labelcolor='#c8cde0')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with tab2:
        fig,ax=plt.subplots(figsize=(10,7)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#161b27')
        for i,(name,vals) in enumerate(results.items()):
            fpr,tpr,_=roc_curve(y_test,vals['y_prob'])
            ax.plot(fpr,tpr,lw=2.5,color=COLORS[i],label=f"{name} (AUC={vals['AUC-ROC']:.3f})")
        ax.plot([0,1],[0,1],'--',color='#3a4160',lw=1.5,label='Aléatoire')
        ax.set_xlabel('FPR',color='#6b7594'); ax.set_ylabel('TPR',color='#6b7594')
        ax.set_title('Courbes ROC',color='#c8cde0',fontsize=13,fontweight='bold')
        ax.legend(loc='lower right',fontsize=9,framealpha=0.1,facecolor='#161b27',labelcolor='#c8cde0')
        ax.spines['bottom'].set_color('#1e2740'); ax.spines['left'].set_color('#1e2740')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with tab3:
        sel=st.selectbox("Modèle",list(results.keys()))
        cm=confusion_matrix(y_test,results[sel]['y_pred'])
        fig,ax=plt.subplots(figsize=(6,5)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#161b27')
        sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',ax=ax,xticklabels=['Sain','Maladie'],
                    yticklabels=['Sain','Maladie'],linewidths=2,linecolor='#0f1117',annot_kws={'size':14,'weight':'bold'})
        ax.set_title(f'{sel}',color='#c8cde0',fontsize=12,fontweight='bold',pad=12)
        ax.set_xlabel('Prédit',color='#6b7594'); ax.set_ylabel('Réel',color='#6b7594')
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif "Prédiction" in page:
    st.markdown('<div class="hero-title" style="font-family:\'DM Serif Display\',serif;color:#fff;font-size:2rem;margin-bottom:0.5rem;">Prédiction <span style="color:#ef5350;">Patient</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7594;margin-bottom:1.5rem;font-size:14px;">Renseignez les paramètres cliniques pour obtenir une prédiction.</div>', unsafe_allow_html=True)
    col_m,_=st.columns([1,2])
    with col_m: model_choice=st.selectbox("Algorithme",list(results.keys()),index=1)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("**Informations générales**")
        age=st.slider("Âge",29,77,55)
        sex_opt=st.radio("Sexe",["Femme","Homme"]); sex_val=1 if sex_opt=="Homme" else 0
        cp=st.selectbox("Douleur thoracique (cp)",[0,1,2,3],format_func=lambda x:f"Type {x}")
        trestbps=st.slider("Pression artérielle (mmHg)",94,200,130)
        chol=st.slider("Cholestérol (mg/dl)",126,564,246)
    with c2:
        st.markdown("**Paramètres cardiaques**")
        fbs_opt=st.radio("Glycémie à jeun > 120",["Non","Oui"]); fbs_val=1 if fbs_opt=="Oui" else 0
        restecg=st.selectbox("ECG au repos",[0,1,2])
        thalach=st.slider("Fréquence cardiaque max",71,202,150)
        exang_opt=st.radio("Angine à l'effort",["Non","Oui"]); exang_val=1 if exang_opt=="Oui" else 0
    with c3:
        st.markdown("**Paramètres complémentaires**")
        oldpeak=st.slider("Dépression ST",0.0,6.2,1.0,step=0.1)
        slope=st.selectbox("Pente ST",[0,1,2])
        ca=st.selectbox("Vaisseaux (ca)",[0,1,2,3])
        thal=st.selectbox("Thalassémie",[1,2,3],format_func=lambda x:{1:"Normal",2:"Défaut fixé",3:"Défaut réversible"}[x])
    st.markdown("---")
    if st.button("🔍  Analyser ce patient"):
        patient=np.array([[age,sex_val,cp,trestbps,chol,fbs_val,restecg,thalach,exang_val,oldpeak,slope,ca,thal]])
        patient_s=scaler.transform(patient)
        model=results[model_choice]['model']
        pred=model.predict(patient_s)[0]; prob=model.predict_proba(patient_s)[0]
        col_r,col_p=st.columns([1,1])
        with col_r:
            if pred==1:
                st.markdown(f"""<div class="result-danger">
                    <div style="font-size:2rem;">❤️‍🩹</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#ef8a87;">Maladie détectée</div>
                    <div style="font-size:3rem;font-weight:600;color:#ef5350;">{prob[1]:.1%}</div>
                    <div style="color:#6b7594;font-size:13px;">probabilité de maladie</div>
                    <div style="color:#3a4160;font-size:12px;margin-top:8px;">Modèle : {model_choice}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-safe">
                    <div style="font-size:2rem;">💚</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#80cbc4;">Pas de maladie</div>
                    <div style="font-size:3rem;font-weight:600;color:#26a69a;">{prob[0]:.1%}</div>
                    <div style="color:#6b7594;font-size:13px;">probabilité d'être sain</div>
                    <div style="color:#3a4160;font-size:12px;margin-top:8px;">Modèle : {model_choice}</div>
                </div>""", unsafe_allow_html=True)
        with col_p:
            fig,ax=plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#161b27'); ax.set_facecolor('#161b27')
            ax.barh(['Sain','Maladie'],[prob[0],prob[1]],color=['#26a69a','#ef5350'],alpha=0.9,height=0.5)
            ax.set_xlim(0,1); ax.set_xlabel('Probabilité',color='#6b7594')
            ax.set_title('Probabilités',color='#c8cde0',fontsize=11,fontweight='bold')
            for i,p in enumerate([prob[0],prob[1]]):
                ax.text(p+0.02,i,f'{p:.1%}',va='center',color='#c8cde0',fontweight='bold',fontsize=13)
            ax.spines['bottom'].set_color('#1e2740'); ax.spines['left'].set_color('#1e2740')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("""<div style="background:#161b27;border:1px solid #1e2740;border-radius:10px;padding:12px 16px;margin-top:1rem;color:#6b7594;font-size:13px;">
            ⚠️ Cette prédiction est à titre informatif uniquement. Consultez un médecin qualifié pour tout diagnostic médical.
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Comparaison tous les modèles</div>', unsafe_allow_html=True)
        all_preds=[{'Modèle':name,'Prédiction':'🔴 Maladie' if results[name]['model'].predict(patient_s)[0]==1 else '🟢 Sain',
            'Prob. Maladie':f'{results[name]["model"].predict_proba(patient_s)[0][1]:.1%}',
            'F1-Score':f'{vals["F1-Score"]:.4f}'} for name,vals in results.items()]
        st.dataframe(pd.DataFrame(all_preds),use_container_width=True,hide_index=True)