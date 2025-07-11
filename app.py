
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


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@quantuniversity.com](mailto:info@quantuniversity.com)
''')
