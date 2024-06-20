import sqlite3
import smtplib
import os
from email.message import EmailMessage


# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('shop.db')
cursor = conn.cursor()

email_contact = cursor.execute("SELECT email FROM contacts WHERE name = 'Bob'").fetchone()[0] 
print(email_contact)
revenue = cursor.execute("SELECT revenue FROM sales WHERE date = '2023-01-01'").fetchone()[0]
revenue2 = cursor.execute("SELECT revenue FROM sales WHERE date = '2023-01-02'").fetchone()[0]
print(revenue)
print(revenue2)

tot = revenue + revenue2

# SMTP server configuration
smtp_server = 'smtp.gmail.com'
smtp_port = 587
username = 'dataintensive.project@gmail.com'
password = os.getenv('EMAIL_PASSWORD')

# Email content
from_email = 'dataintensive.project@gmail.com'
to_email = 'manual_test@example.com'
subject = 'Test Email'
body = tot

msg = EmailMessage()
msg.set_content(str(body))
msg['Subject'] = subject
msg['From'] = from_email
msg['To'] = to_email

try:
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(from_email, password)
        smtp_server.sendmail(from_email, to_email, msg.as_string())
        print("Message sent!")

except Exception as e:
    print(f'Error sending email: {e}')

conn.close()