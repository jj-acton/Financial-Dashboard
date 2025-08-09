import streamlit as st
import pymysql

KEY = st.secrets["MySQL"]["key"]

conn = pymysql.connect(
    host='database-1-finacial-dashboard-contact-page.ctkey4424y8v.eu-west-2.rds.amazonaws.com',
    user='admin',
    password= KEY,
    db='database-1-finacial-dashboard-contact-page',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cursor:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            company VARCHAR(255),
            message TEXT
        )
    """)
    conn.commit()

with st.form(key="contact_form" ,clear_on_submit=True, enter_to_submit=False):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    company = st.text_input("Company")
    message = st.text_input("Message/Feedback")
    
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        with conn.cursor() as cursor:
            sql = "INSERT INTO contacts (name, email, company, message) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (name, email, company, message))
            conn.commit()
        st.success("Your message has been sent. Thank you!")

conn.close()