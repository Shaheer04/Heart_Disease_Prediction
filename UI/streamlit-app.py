import streamlit as st

pg = st.navigation([
    st.Page("components/about.py", title="About", icon=":material/favorite:"),
    st.Page("components/main.py", title="Heart disease Prediction", icon="🔥"),
    #st.Page("monitor.py", title="Monitor", icon=":material/visibility:")
])
if __name__ == "__main__":
    pg.run()