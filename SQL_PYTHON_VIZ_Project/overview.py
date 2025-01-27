import streamlit as st

variable_info = st.session_state["variable_info"]
df = st.session_state["df"]
# st.html('''<script src="https://kit.fontawesome.com/cea37793b2.js" crossorigin="anonymous"></script>''')

st.write('#### **Analysis Author:** Henrique M. L. Pereira')
# https://michaelcurrin.github.io/badge-generator/#/
st.markdown('''
[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/HenrikPereira)  
[![Gmail - Pessoal](https://img.shields.io/badge/Gmail-Pessoal-2ea44f?logo=gmail)](mailto:henriquemiguel.pereira@gmail.com)  
[![FC - Profissional](https://img.shields.io/badge/FC-Profissional-2ea44f?logo=maildotru)](mailto:henrique.pereira@fundacaochampalimaud.pt.com)  
[![FMUP - Institucional](https://img.shields.io/badge/FMUP-Institucional-2ea44f)](mailto:up2024000000@edu.fm.ul.pt)
''')

st.markdown('##### CLASS: **INFORM**')
st.markdown("***PhD programme HEADS** of **FMUP** in collaboration with **CINTESIS***")
st.markdown('## Introduction')

st.markdown("""
- Objectives of the present work:
    - Demonstrate the capabilities of streamlit on a Health Data Science project.
    - Make use of the Querying Language SQL using DuckDB.
    - Demonstrate the use of visualizations within streamlit.
- Data Source:
    - ***Diagnostic Wisconsin Breast Cancer Database*** from [UCI ml repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
    - According to the [Original Paper](https://www.semanticscholar.org/paper/Nuclear-feature-extraction-for-breast-tumor-Street-Wolberg/53f0fbb425bc14468eb3bf96b2e1d41ba8087f36),
the variables we are about to explore, were computed in a very particular methodology which I strongly advise to at least have a look at to understand the context of my decisions along this work.
""")

st.markdown('## Variables')

st.markdown(f"Besides the variables `ID` and `Diagnosis`, {variable_info.split('3-32)')[-1]}")
st.markdown('For each of the last variables ( a) to j) ), the **mean value** (suffix `1`), '
            '**largest (or *worst*) value** (suffix `2`), and **standard error** (suffix `3`) '
            'were found over the whole range of isolated cells')

st.info("For further details and insights, see pages of Exploratory Data Analysis (EDA) and Models on the sidebar.")
