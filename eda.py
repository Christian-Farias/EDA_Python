import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9
})

st.set_page_config(page_title="EDA Automática", layout="wide")

st.title("Análise Exploratória de Dados (EDA) de Planilhas")
st.write("Faça upload de uma planilha para realizar a análise exploratória automaticamente.")

uploaded_file = st.file_uploader("Faça o upload do arquivo", type=["csv", "xlsx"])

if uploaded_file:
    
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    if df.empty:
        st.warning("O arquivo está vazio.")
        st.stop()

    st.subheader("Prévia dos Dados")
    st.dataframe(df.head(5))

    st.subheader("Diagnóstico das Colunas")

    col_info = []

    for col in df.columns:
        col_info.append({
            "Coluna": col,
            "Tipo de Dado": str(df[col].dtype),
            "% Nulos": round((df[col].isna().sum() / df.shape[0]) * 100, 2),
            "Valores Únicos": df[col].nunique()
        })

    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df)

    st.divider()

    numerical_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    cat_cols = [
        col for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) 
        or df[col].dtype.name == "category"
    ]

    
    # Abas
    tab1, tab2 = st.tabs(["Variáveis Numéricas", "Variáveis Categóricas"])

    # Análise Numérica
    with tab1:
        st.subheader("Análise Exploratória - Variáveis Numéricas")

        if not numerical_cols:
            st.info("Nenhuma variável numérica encontrada.")
        else:
            for col in numerical_cols:
                st.markdown(f"### {col}")

                series = df[col].dropna()

                if series.empty:
                    st.warning("Coluna sem valores numéricos válidos.")
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                limite_inferior = q1 - 1.5 * iqr
                limite_superior = q3 + 1.5 * iqr
                media = series.mean()
                mediana = series.median()
                desvio_padrao = series.std()

                outliers = series[(series < limite_inferior) | (series > limite_superior)]

                col1, col2, col3 = st.columns(3)

                col1.metric("Q1", round(q1, 2))
                col2.metric("Q3", round(q3, 2))
                col3.metric("IQR", round(iqr, 2))

                col1.metric("Limite Inferior", round(limite_inferior, 2))
                col2.metric("Limite Superior", round(limite_superior, 2))
                col3.metric("Outliers", outliers.count())

                col1.metric("Média", round(media, 2))
                col2.metric("Mediana", round(mediana, 2))
                col3.metric("Desvio Padrão", round(desvio_padrao, 2))

                fig, ax = plt.subplots(1, 2, figsize=(8, 3))

                ax[0].hist(series, bins=25, edgecolor="white")
                ax[0].set_title("Distribuição", fontweight="bold")
                ax[0].set_xlabel(col)
                ax[0].grid(alpha=0.2)

                ax[1].boxplot(series, vert=False, patch_artist=True)
                ax[1].set_title("Boxplot", fontweight="bold")
                ax[1].axvline(limite_inferior, linestyle="--", linewidth=1)
                ax[1].axvline(limite_superior, linestyle="--", linewidth=1)
                ax[1].grid(alpha=0.2)

                plt.tight_layout()
                st.pyplot(fig)
                st.divider()

    # Análise Categórica
    with tab2:
        st.subheader("Análise Exploratória - Variáveis Categóricas")

        if not cat_cols:
            st.info("Nenhuma variável categórica encontrada.")
        else:
            for col in cat_cols:
                st.markdown(f"### {col}")

                series = df[col].dropna().astype(str)

                if series.empty:
                    st.warning("Coluna sem valores categóricos válidos.")
                    continue

                vc = series.value_counts()
                n_categorias = vc.shape[0]

                st.write(f"Categorias únicas: **{n_categorias}**")

                if n_categorias <= 5:
                    plot_data = vc
                    st.dataframe(vc)
                else:
                    top4 = vc.head(4)
                    outros = vc.iloc[4:].sum()
                    plot_data = pd.concat([top4, pd.Series({"Outros": outros})])

                plot_data = plot_data.sort_values()

                fig, ax = plt.subplots(figsize=(6, 3))

                plot_data.plot(kind="barh", ax=ax)

                ax.set_title(f"Distribuição de {col}", fontweight="bold")
                ax.set_xlabel("Quantidade")
                ax.grid(axis="x", alpha=0.2)

                plt.tight_layout()
                st.pyplot(fig)

                st.divider()

    
