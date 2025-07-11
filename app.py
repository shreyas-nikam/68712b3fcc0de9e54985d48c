
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we explore operational risk modeling concepts, bridging the gap between theoretical understanding and practical application.
We'll define hypothetical scenarios, simulate Internal Loss Data (ILD), and combine these data sources using statistical averaging techniques.
A key objective is to visually demonstrate the counter-intuitive 'stability paradox' in risk aggregation.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Scenario Fitting", "ILD Generation and Fitting", "Distribution Combination and Paradox Simulation"])
if page == "Scenario Fitting":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "ILD Generation and Fitting":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Distribution Combination and Paradox Simulation":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
