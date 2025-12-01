import pandas as pd
import mysql.connector
import json

# -----------------------------
# MySQL CONNECTION
# -----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="abdul@123",
    database="job_application_form_db"
)
cursor = db.cursor()

# -----------------------------
# READ EXCEL
# -----------------------------
file_path = r"D:/Application_form/Interview_MCQ_By_Experience_12to18_Per_Range.xlsx"

xls = pd.ExcelFile(file_path)
print("Sheets found:", xls.sheet_names)

required_columns = [
    "Question",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "Correct Answer (A/B/C/D)",
    "Experience Range (Years)"
]

# -----------------------------
# PROCESS EACH SHEET
# -----------------------------
for sheet in xls.sheet_names:

    print(f"\nProcessing sheet: {sheet}")

    df = pd.read_excel(xls, sheet_name=sheet)

    # Check missing columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"❌ Skipped sheet '{sheet}' — missing columns: {missing}")
        continue

    # Iterate rows
    for _, row in df.iterrows():

        question_text = str(row["Question"]).strip()
        option_a = str(row["Option A"]).strip()
        option_b = str(row["Option B"]).strip()
        option_c = str(row["Option C"]).strip()
        option_d = str(row["Option D"]).strip()
        correct = str(row["Correct Answer (A/B/C/D)"]).strip().upper()
        experience = str(row["Experience Range (Years)"]).strip()

        if correct not in ["A", "B", "C", "D"]:
            print(f"⚠ Skipped row with invalid correct option: {correct}")
            continue

        mcq_options_json = json.dumps({
            "A": option_a,
            "B": option_b,
            "C": option_c,
            "D": option_d
        }, ensure_ascii=False)

        # Insert into DB
        sql = """
            INSERT INTO questions 
            (question_text, question_type, mcq_options, correct_option, model_answer,
             skill_tag, experience_tag, created_by)
            VALUES (%s, 'mcq', %s, %s, NULL, %s, %s, 'system')
        """

        cursor.execute(sql, (
            question_text,
            mcq_options_json,
            correct,
            sheet,        # skill_tag from sheet name
            experience    # experience_tag
        ))

    db.commit()
    print(f"✔ Sheet '{sheet}' inserted successfully!")

print("\n🎉 ALL DONE — All questions inserted into MySQL!")
cursor.close()
db.close()
