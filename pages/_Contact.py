import streamlit as st
import pymysql
import boto3
from botocore.exceptions import ClientError

st.set_page_config(
    page_title="Contact me",
    layout= 'wide',
    page_icon="📧",
    initial_sidebar_state="auto"
)

PASSWORD = st.secrets["MySQL"]["password"]
HOST = st.secrets["MySQL"]["host"]
USER = st.secrets["MySQL"]["user"]
DB = st.secrets["MySQL"]["db"]
SNS_TOPIC_ARN = st.secrets["SNS"]["topic_arn"]  

def get_connection():
    return pymysql.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        db=DB,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor #result is returned as dict
    )

sns_client = boto3.client('sns', region_name='eu-west-2') 

def send_sns_notification(subject, message):
    try:
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        return response
    except ClientError as e:
        st.error(f"Failed to send notification: {e}")
        return None

with get_connection() as conn:
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

with st.form(key="contact_form", clear_on_submit=True, enter_to_submit=False):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    company = st.text_input("Company")
    message = st.text_area("Message/Feedback")  

    submitted = st.form_submit_button("Submit")

    if submitted:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                sql = "INSERT INTO contacts (name, email, company, message) VALUES (%s, %s, %s, %s)"#placeholder
                cursor.execute(sql, (name, email, company, message))
                conn.commit()
        subject = f"New Contact Form Submission from {name}"
        body = f"Name: {name}\nEmail: {email}\nCompany: {company}\nMessage:\n{message}"
        send_sns_notification(subject, body)

        st.success("Your message has been sent and notification triggered. Thank you!")