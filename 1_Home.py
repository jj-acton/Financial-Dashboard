import streamlit as st

st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

DIVIDER_COLOUR = "blue"

st.title("Welcome to the Home Page")
st.header("John Acton", divider=f'{DIVIDER_COLOUR}')
st.markdown("""
<div style="text-align: justify;">
Welcome to the home page of my financial portfolio. Here you can find various financial analyses and visualisations. These analyses are projects I have worked on, showcasing my skills in data analysis and visualisation using Python libraries such as Pandas, Numpy, Matplotlib, Scikit-learn, Yahoo Finance and Streamlit as well as tools such as SQL. These projects include stock market analysis, financial forecasting, machine learning and portfolio management strategies. Each project is designed to provide insights into financial data and users understanding of market trends. Feel free to explore the different sections and projects available in this portfolio. If you have any questions or feedback, please reach out to me through the contact page. If you are having issues viewing the page or it does not look correct you can press the three dots at the top of the screen, go to settings and turn on/off wide mode.
</div>
""",unsafe_allow_html=True)

st.subheader("About Me")
col1, col2= st.columns(2, gap="large", vertical_alignment="center")
with col1:
    st.markdown("""
    <div style="text-align: justify;">
    My name is John Acton, but I usually go by JJ. I’m a recent graduate from Imperial College London with a postgraduate research master’s in Physics, where I focused on discovering new materials for next-generation solar cells. During my thesis, I developed a strong interest in data analysis and visualisation, which sparked my move toward financial analysis. Since then, I’ve worked on projects that combine my passion for data with my interest in finance. This web app is a way for me to stand out in roles that focus on data-driven analytics and market research. Outside of work, I enjoy skiing, scuba diving, travelling, cooking, and music.
""", unsafe_allow_html=True)
    
with col2:
    st.image("images/Profile_picture.png", width=350)

st.subheader("My Thesis")
st.markdown("""
            <div style="text-align: justify;">
            Below is a link to download my thesis, which details my research on the discovery of new materials for next generation solar cells. It includes my findings, methodologies, and conclusions drawn from the research conducted during my postgraduate studies at Imperial College London. I am proud of this work and although it is not directly related to finance, it showcases my analytical skills and ability to conduct thorough research. I believe these skills are transferable to financial analysis and data-driven decision making.
            </div>
            """, unsafe_allow_html=True)
st.subheader("")
with open("images/JJ_Acton_Thesis.pdf", "rb") as file:
        st.download_button(
            label="Download Thesis",
            data=file,
            file_name="JJ_Acton_Thesis.pdf",
            mime="application/pdf",
            icon="➡️"
        )