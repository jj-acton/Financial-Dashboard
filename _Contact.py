import streamlit as st
import sqlite3

st.set_page_config(
    page_title="Contact",
    page_icon="📧",
    initial_sidebar_state="auto",
    layout="wide"
)

st.title("Contact Us")

conn = sqlite3.connect("contact_data.db")
c = conn.cursor()

c.execute(
    """
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        company TEXT,
        message TEXT
    )
    """
)
conn.commit()

with st.form(key="contact_form" ,clear_on_submit=True, enter_to_submit=False):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    company = st.text_input("Company")
    message = st.text_input("Message/Feedback")
    
    submitted = st.form_submit_button("Submit")
    
    if submitted : 
        c.execute("INSERT INTO contacts (name, email, company, message) VALUES (?, ?, ?, ?)", (name, email, company, message))
        conn.commit() 
        st.success("Your message has been sent. Thank You!")

conn.close()