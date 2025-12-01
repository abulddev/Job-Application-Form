import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="abdul@123",
        database="job_application_form_db",
        raise_on_warnings=True
    )
