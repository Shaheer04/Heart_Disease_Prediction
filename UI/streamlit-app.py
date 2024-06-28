import streamlit as st

pg = st.navigation([
    st.Page("about.py", title="About", icon=":material/favorite:"),
    st.Page("main.py", title="Heart disease Prediction", icon="🔥"),
    #st.Page("monitor.py", title="Monitor", icon=":material/visibility:")
])
pg.run()