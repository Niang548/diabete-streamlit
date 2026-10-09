import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Risque de diabète", page_icon="🩺")

st.title("Estimation du risque de diabète")
st.caption("Outil d'aide au dépistage. Il ne remplace pas un avis médical.")

m = joblib.load("modele_diabete.joblib")
cols = m["colonnes"]
seuil = m["seuil"]


def champ(label, mini, maxi, defaut, cle, optionnel=False, step=None):
    """Champ numérique, avec une case 'inconnu' si optionnel."""
    inconnu = False
    if optionnel:
        inconnu = st.checkbox("Valeur inconnue", key="inc_" + cle)
    options = dict(min_value=mini, max_value=maxi, value=defaut, key=cle, disabled=inconnu)
    if step is not None:
        options["step"] = step
    valeur = st.number_input(label, **options)
    return float("nan") if inconnu else valeur


col1, col2 = st.columns(2)

with col1:
    pregnancies = champ("Nombre de grossesses", 0, 20, 1, "preg")
    glucose = champ("Glucose (mg/dL)", 40, 250, 117, "glu", optionnel=True)
    blood_pressure = champ("Tension diastolique (mm Hg)", 20, 130, 72, "bp", optionnel=True)
    skin = champ("Épaisseur du pli cutané (mm)", 5, 100, 29, "skin", optionnel=True)

with col2:
    insulin = champ("Insuline (µU/mL)", 10, 900, 125, "ins", optionnel=True)
    bmi = champ("IMC (kg/m²)", 15.0, 70.0, 32.0, "bmi", optionnel=True, step=0.1)
    pedigree = champ("Antécédents familiaux (score)", 0.05, 2.5, 0.37, "ped", step=0.01)
    age = champ("Âge (ans)", 15, 100, 29, "age")

if st.button("Prédire", type="primary"):
    donnees = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }])[cols]

    nb_inconnues = int(donnees.isna().sum().sum())

    # même préparation que pendant l'entraînement
    x_imp = pd.DataFrame(m["imputer"].transform(donnees), columns=cols)
    x_sc = pd.DataFrame(m["scaler"].transform(x_imp), columns=cols)
    proba = m["model"].predict_proba(x_sc)[0, 1]
        
    with st.expander("Voir les valeurs utilisées par le modèle"):
        st.dataframe(x_imp.round(2))
        
    st.subheader("Résultat")
    st.metric("Probabilité estimée de diabète", f"{proba:.0%}")
    st.progress(float(proba))

    if proba >= seuil:
        st.warning(
            f"Risque élevé : la probabilité dépasse le seuil de {seuil:.0%}. "
            "Un examen de confirmation (glycémie à jeun, HbA1c) est conseillé."
        )
    else:
        st.success(
            f"Risque faible : la probabilité est sous le seuil de {seuil:.0%}. "
            "Ce résultat ne garantit pas l'absence de diabète."
        )

    if nb_inconnues > 0:
        st.info(
            f"{nb_inconnues} valeur(s) inconnue(s) remplacée(s) par la médiane des données "
            "d'entraînement. Plus il y a de valeurs inconnues, moins l'estimation est fiable."
        )

    st.caption(
        "Le modèle a été entraîné sur 768 patientes d'origine Pima. "
        "Il détecte environ 83 % des cas et produit des fausses alertes : "
        "il ne remplace pas un diagnostic médical."
    )
