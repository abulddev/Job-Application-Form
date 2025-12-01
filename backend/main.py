from fastapi import FastAPI, Form, Request, Response, Cookie, File, UploadFile, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mysql.connector
import asyncio, requests
import random, json, time
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from config import get_db  # your existing DB connection function
from utils import hash_password, verify_password
from passlib.context import CryptContext
import secrets, os, tempfile, zipfile
from passlib.hash import bcrypt
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from datetime import datetime, timedelta, date
import uvicorn, re
import string, random, smtplib
from email.mime.text import MIMEText
from fastapi import UploadFile, BackgroundTasks
from gpt4all import GPT4All
import mimetypes

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Jinja2 templates from frontend folder
templates = Jinja2Templates(directory="D:/Application_form/frontend")

UPLOAD_DIR = r"D:/Application_form/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH = "D:/Application_form/GPT4ALL/models"

llm = GPT4All(
    model_name=MODEL_NAME,
    model_path=MODEL_PATH,
    allow_download=False,
    verbose=False,
)
conf = ConnectionConfig(
    MAIL_USERNAME = "ab.samad.ansaar@gmail.com",
    MAIL_PASSWORD = "lslx mpim arze wdhn",
    MAIL_FROM = "ab.samad.ansaar@gmail.com",
    MAIL_PORT = 587,
    MAIL_SERVER = "smtp.gmail.com",  # e.g. smtp.gmail.com
    MAIL_STARTTLS=True,      # REQUIRED — replaces MAIL_TLS
    MAIL_SSL_TLS=False, 
    USE_CREDENTIALS = True,
    VALIDATE_CERTS = True
)

@app.get("/view-file/{filename}")
def view_file(filename: str):
    file_path = f"uploads/{filename}"

    content_types = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

    ext = filename.split(".")[-1].lower()
    ctype = content_types.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=ctype,
        headers={"Content-Disposition": f'inline; filename="{filename}"'}
    )

    
@app.get("/download-file/{file_name}")
def download_file(file_name: str):
    file_path = f"uploads/{file_name}"

    if not os.path.exists(file_path):
        return Response(content="File not found", status_code=404)

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=file_name,
        headers={"Content-Disposition": "attachment"}
    )
        
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, response: Response, email: str = Form(...), password: str = Form(...)):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM admin_users WHERE email=%s", (email,))
    admin = cursor.fetchone()
    cursor.close()
    db.close()

    if admin and verify_password(password, admin["password"]):
        redirect = RedirectResponse(url="/dashboard", status_code=303)
        redirect.set_cookie(key="admin_email", value=admin["email"])
        return redirect
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid email or password"
        })
    
@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {
        "request": request,
        "error": "",
        "success": ""
    })
    
@app.post("/forgot-password")
async def forgot_password(request: Request, email: str = Form(...)):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM admin_users WHERE email=%s", (email,))
    admin = cursor.fetchone()

    if not admin:
        cursor.close()
        db.close()
        return templates.TemplateResponse("forgot_password.html", {
            "request": request,
            "error": "Email not found",
            "success": ""
        })

    # generate a secure token
    token = secrets.token_urlsafe(32)

    # ⚡ update both token AND creation timestamp
    cursor.execute("""
        UPDATE admin_users 
        SET reset_token=%s, reset_token_created_at=NOW() 
        WHERE email=%s
    """, (token, email))
    db.commit()
    cursor.close()
    db.close()

    reset_link = f"http://127.0.0.1:8000/reset-password?token={token}"

    # Prepare email message
    message = MessageSchema(
        subject="Reset your admin password",
        recipients=[email],
        body=f"Click this link to reset your password: {reset_link}",
        subtype=MessageType.plain
    )

    fm = FastMail(conf)
    await fm.send_message(message)

    return templates.TemplateResponse("forgot_password.html", {
        "request": request,
        "error": "",
        "success": "A password reset link has been sent to your email."
    })

@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(request: Request, token: str):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM admin_users WHERE reset_token=%s", (token,))
    admin = cursor.fetchone()

    cursor.close()
    db.close()

    if not admin:
        return HTMLResponse("<h2>Invalid or expired token</h2>", status_code=400)

    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token": token,
        "error": "",
        "success": ""
    })

@app.post("/reset-password")
def reset_password(request: Request, token: str = Form(...), password: str = Form(...), confirm: str = Form(...)):

    # 1) Password match check
    if password != confirm:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token": token,
            "error": "Passwords do not match",
            "success": ""
        })

    # 2) Password must be >= 6 characters
    if len(password) < 6:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token": token,
            "error": "Password must be at least 6 characters long",
            "success": ""
        })

    db = get_db()
    cursor = db.cursor(dictionary=True)

    # 3) Check token in DB
    cursor.execute("SELECT * FROM admin_users WHERE reset_token=%s", (token,))
    admin = cursor.fetchone()

    if not admin:
        cursor.close()
        db.close()
        return HTMLResponse("<h2>Invalid or expired reset link</h2>", status_code=400)

    # 4) Token expiry check - 1 hour
    created_at = admin["reset_token_created_at"]
    if not created_at or (datetime.utcnow() - created_at > timedelta(hours=24)):
        cursor.close()
        db.close()
        return HTMLResponse("<h2>Reset link expired. Request a new one.</h2>", status_code=400)

    # 5) Old password cannot be reused
    if verify_password(password, admin["password"]):
        cursor.close()
        db.close()
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token": token,
            "error": "New password cannot be the old password",
            "success": ""
        })

    # 6) Hash new password using Argon2
    hashed_pw = hash_password(password)

    # 7) Update DB & clear token
    cursor.execute(
        "UPDATE admin_users SET password=%s, reset_token=NULL, reset_token_created_at=NULL WHERE id=%s",
        (hashed_pw, admin["id"])
    )
    db.commit()

    cursor.close()
    db.close()

    # 8) Success
    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token": token,
        "error": "",
        "success": "Password successfully changed! You can now login."
    })

    
@app.get("/dashboard")
def dashboard(request: Request):
    admin_email = request.cookies.get("admin_email")

    if not admin_email:
        return RedirectResponse("/login")

    admin_initials = admin_email[0].upper()  # Example: "a" → "A"

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "admin_email": admin_email,
            "admin_initials": admin_initials
        }
    )

@app.post("/delete-assignment/{assignment_id}")
def delete_assignment(assignment_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        # 1) Get test_id from assignment
        cursor.execute("SELECT test_id FROM assignments WHERE id = %s", (assignment_id,))
        row = cursor.fetchone()

        if row and row["test_id"]:
            test_id = row["test_id"]

            # Delete all submissions for this test
            cursor.execute("DELETE FROM test_submissions WHERE test_id = %s", (test_id,))

            # Delete the test itself
            cursor.execute("DELETE FROM tests WHERE id = %s", (test_id,))

        # 2) Delete assignment itself
        cursor.execute("DELETE FROM assignments WHERE id = %s", (assignment_id,))

        db.commit()

    except Exception as e:
        print("Delete Assignment Error:", e)

    finally:
        cursor.close()
        db.close()

    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/delete-test/{test_id}")
def delete_test(test_id: int):
    db = get_db()
    cursor = db.cursor()

    # Remove candidate submissions
    cursor.execute("DELETE FROM test_submissions WHERE id = %s", (test_id,))
   
    # Remove the test itself (if stored in tests table)
    cursor.execute("DELETE FROM tests WHERE id = %s", (test_id,))

    db.commit()
    cursor.close()
    db.close()

    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/delete-test-submission/{test_id}")
def delete_test_submission(test_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        # Delete submissions
        cursor.execute("DELETE FROM test_submissions WHERE test_id = %s", (test_id,))

        # Delete test (correct column is id, NOT test_id)
        cursor.execute("DELETE FROM tests WHERE id = %s", (test_id,))

        # Also remove assignment linked to this test
        cursor.execute("DELETE FROM test_submissions WHERE test_id = %s", (test_id,))

        db.commit()

    except Exception as e:
        print("Delete Test Error:", e)

    finally:
        cursor.close()
        db.close()

    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/delete-assignment-list/{assignment_id}")
def delete_assignment_list(assignment_id: int):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM assignment_master WHERE id = %s", (assignment_id,))
        db.commit()
        cursor.close()
        db.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

  
# --- Generate temporary password ---
def generate_temp_password(length=10):
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choice(chars) for _ in range(length))

@app.get("/assignments", response_class=HTMLResponse)
def assignments_page(request: Request):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM assignment_master ORDER BY id DESC")
    assignments = cursor.fetchall()

    cursor.close()
    db.close()

    return templates.TemplateResponse("assignments.html", {
        "request": request,
        "assignments": assignments
    })


# --- Pages ---
@app.get("/create-assignment", response_class=HTMLResponse)
def create_assignment_page(request: Request):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Load skill tags
    cursor.execute("SELECT DISTINCT skill_tag FROM questions ORDER BY skill_tag ASC")
    skills = [row["skill_tag"] for row in cursor.fetchall()]

    # Load experience tags
    cursor.execute("SELECT DISTINCT experience_tag FROM questions ORDER BY experience_tag ASC")
    exps = [row["experience_tag"] for row in cursor.fetchall()]

    cursor.close()
    db.close()

    return templates.TemplateResponse("create_assignment.html", {
        "request": request,
        "skills": skills,
        "experiences": exps,
        "error": "",
        "success": "",
        "current_date": date.today().isoformat()
    })


@app.post("/create-assignment", response_class=HTMLResponse)
async def create_assignment(
    request: Request,
    assignment_name: str = Form(...),
    scenario_text: list[str] = Form(...),        # multiple text blocks
    skill_tag: str = Form(...),
    experience_tag: str = Form(...),
    due_date: str = Form(...),
    attachments: list[UploadFile] = File(None)   # multiple optional files
):
    try:
        saved_pairs = []  # list of {text: , file: }

        # Ensure attachments length matches text length, fill missing with None
        if not attachments:
            attachments = [None] * len(scenario_text)
        else:
            while len(attachments) < len(scenario_text):
                attachments.append(None)

        # ---------- Save Files + Pair with Text ----------
        for idx, text in enumerate(scenario_text):
            file = attachments[idx]
            saved_file = None

            if file and file.filename:
                ext = file.filename.split(".")[-1].lower()
                allowed = ["jpg","jpeg","png","xlsx","xls","doc","docx","ppt","pptx","pdf"]

                if ext not in allowed:
                    return templates.TemplateResponse("create_assignment.html", {
                        "request": request,
                        "error": f"File type not allowed: {file.filename}",
                        "success": ""
                    })

                contents = await file.read()
                file_path = os.path.join("uploads", file.filename)
                with open(file_path, "wb") as f:
                    f.write(contents)

                saved_file = file.filename

            saved_pairs.append({"text": text, "file": saved_file})

        # Save JSON in DB
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            INSERT INTO assignment_master
            (assignment_name, scenario_text, skill_tag, experience_tag, due_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            assignment_name,
            json.dumps(saved_pairs),   # ← store structured data
            skill_tag,
            experience_tag,
            due_date
        ))

        db.commit()
        cursor.close()
        db.close()
        
        if datetime.strptime(due_date, "%Y-%m-%d").date() < date.today():
          return templates.TemplateResponse("create_assignment.html", {
            "request": request,
            "error": "Due date cannot be in the past!",
            "success": ""
       })
        return templates.TemplateResponse("create_assignment.html", {
            "request": request,
            "success": "Assignment created successfully!",
            "error": ""
        })

    except Exception as e:
        return templates.TemplateResponse("create_assignment.html", {
            "request": request,
            "error": str(e),
            "success": ""
        })

@app.post("/admin/send-assignment")
async def send_assignment(request: Request):
    form = await request.form()
    assignment_id = form.get("assignment_id")  # hidden field

    if not assignment_id:
        return {"success": False, "error": "Missing assignment_id"}

    # --- DB Connection ---
    db = get_db()
    cursor = db.cursor(dictionary=True)

    candidates = []

    # Read max 20 candidates
    for i in range(1, 21):
        name = form.get(f"name{i}")
        email = form.get(f"email{i}")

        if email and email.strip():
            candidates.append({
                "name": name.strip() if name else "",
                "email": email.strip()
            })

    sent_to = []

    for c in candidates:
        candidate_email = c["email"]
        candidate_name = c["name"] if c["name"] else "Candidate"

        # temp password
        temp_password = secrets.token_hex(3)

        # --------------------------------------------
        # INSERT into assignment_candidates (FIXED)
        # --------------------------------------------
        cursor.execute("""
            INSERT INTO assignment_candidates 
                (assignment_id, candidate_name, candidate_email, temp_password)
            VALUES (%s, %s, %s, %s)
        """, (assignment_id, candidate_name, candidate_email, temp_password))
        db.commit()

        # --------------------------------------------
        # Build login link
        # --------------------------------------------
        assignment_link = (
            f"http://localhost:8000/candidate-login?"
            f"assignment_id={assignment_id}&email={candidate_email}&temp_password={temp_password}"
        )

        subject = "Your Assignment is Ready"

        message = f"""
Hello {candidate_name},

You have been assigned a task.

Click below to view your assignment:
{assignment_link}

Best regards,
Recruitment Team
"""

        await send_email(candidate_email, subject, message)
        sent_to.append(candidate_email)

    cursor.close()
    db.close()

    return {
        "success": True,
        "sent_to": sent_to
    }


        
@app.get("/candidate-login", response_class=HTMLResponse)
def candidate_login_page(request: Request):

    email = request.query_params.get("email", "")
    temp_password = request.query_params.get("temp_password", "")
    assignment_id = request.query_params.get("assignment_id", "")
    test_id = request.query_params.get("test_id", "")

    return templates.TemplateResponse("candidate_login.html", {
        "request": request,
        "email": email,
        "temp_password": temp_password,
        "assignment_id": assignment_id,
        "test_id": test_id,
        "error": ""
    })


@app.post("/candidate-login", response_class=HTMLResponse)
async def candidate_login(request: Request):

    form = await request.form()

    email = form.get("email", "")
    temp_password = form.get("temp_password", "")
    assignment_id = form.get("assignment_id", "")
    test_id = form.get("test_id", "")

    # ----------------------------
    # CASE 1: TEST LOGIN
    # ----------------------------
    if test_id:
        response = RedirectResponse(
            url=f"/test/start/{test_id}",
            status_code=302
        )
        response.set_cookie("candidate_email", email)
        return response

    # ----------------------------
    # CASE 2: ASSIGNMENT LOGIN
    # ----------------------------
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT * FROM assignment_candidates
        WHERE candidate_email=%s AND temp_password=%s AND assignment_id=%s LIMIT 1
        """,
        (email, temp_password, assignment_id)
    )

    candidate = cursor.fetchone()

    cursor.close()
    db.close()

    # ----------------------------
    # INVALID CREDENTIALS
    # ----------------------------
    if not candidate:
        return templates.TemplateResponse("candidate_login.html", {
            "request": request,
            "error": "Invalid email or temporary password.",
            "email": email,
            "temp_password": temp_password,
            "assignment_id": assignment_id,
            "test_id": test_id
        })

    # ----------------------------
    # VALID — REDIRECT TO ASSIGNMENT PAGE
    # ----------------------------
    assignment_id = candidate["assignment_id"]

    response = RedirectResponse(
        url=f"/assignment/{assignment_id}?email={email}",
        status_code=302
    )

    response.set_cookie("candidate_email", email)
    return response


@app.get("/assignment/{assignment_id}", response_class=HTMLResponse)
def candidate_assignment(request: Request, assignment_id: int):

    candidate_email = request.cookies.get("candidate_email")
    if not candidate_email:
        return RedirectResponse("/candidate-login", status_code=302)

    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Validate access
    cursor.execute("""
        SELECT 1 FROM assignment_candidates
        WHERE assignment_id = %s AND candidate_email = %s LIMIT 1
    """, (assignment_id, candidate_email))

    if not cursor.fetchone():
        return HTMLResponse("Assignment not found or not assigned to you.", status_code=404)

    # Assignment
    cursor.execute("SELECT * FROM assignment_master WHERE id = %s", (assignment_id,))
    assignment = cursor.fetchone()

    # Draft / submission
    cursor.execute("""
        SELECT explanation, file_path, is_submitted
        FROM assignment_answers
        WHERE assignment_id = %s AND candidate_email = %s
        LIMIT 1
    """, (assignment_id, candidate_email))

    saved = cursor.fetchone()

    saved_explanation = saved["explanation"] if saved else ""
    saved_file_raw = saved["file_path"] if saved else ""
    is_submitted = saved["is_submitted"] if saved else 0

    # Extract actual filename
    saved_file = os.path.basename(saved_file_raw) if saved_file_raw else ""

    # Parse scenario text JSON
    try:
        parsed = json.loads(assignment["scenario_text"])
        scenario_blocks = parsed if isinstance(parsed, list) else [{"text": assignment["scenario_text"]}]
    except:
        scenario_blocks = [{"text": assignment["scenario_text"]}]

    # Clean filenames inside scenario blocks
    for block in scenario_blocks:
        if "file" in block and block["file"]:
            block["file"] = os.path.basename(block["file"])

    # ---- BASE URL NEEDED FOR GOOGLE VIEWER ----
    base_url = str(request.base_url).rstrip("/")

    return templates.TemplateResponse(
        "candidate_assignment.html",
        {
            "request": request,
            "assignment": assignment,
            "scenario_blocks": scenario_blocks,
            "saved_explanation": saved_explanation,
            "saved_file": saved_file,
            "is_submitted": is_submitted,
            "base_url": base_url,
        }
    )

# ----------------------------------------------------------
# SAVE DRAFT (DB stores ONLY filename)
# ----------------------------------------------------------
@app.post("/assignment/{assignment_id}/save")
async def save_draft(request: Request, assignment_id: int):
    candidate_email = request.cookies.get("candidate_email")
    form = await request.form()

    explanation = form.get("explanation")
    uploaded_file = form.get("file")

    db_filename = None

    if uploaded_file and uploaded_file.filename:
        ext = os.path.splitext(uploaded_file.filename)[1]
        safe_name = f"{assignment_id}_{int(time.time())}{ext}"     # CLEAN filename only

        save_path = os.path.join(UPLOAD_DIR, safe_name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.file.read())

        db_filename = safe_name      # ONLY filename saved in DB

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO assignment_answers (assignment_id, candidate_email, explanation, file_path)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            explanation=%s,
            file_path=COALESCE(%s, file_path)
    """, (assignment_id, candidate_email, explanation, db_filename, explanation, db_filename))

    db.commit()
    cursor.close()
    db.close()

    return {"success": True}


# ----------------------------------------------------------
# SUBMIT FINAL (same clean filename logic)
# ----------------------------------------------------------
@app.post("/assignment/{assignment_id}/submit")
async def submit_assignment(request: Request, assignment_id: int):
    candidate_email = request.cookies.get("candidate_email")
    form = await request.form()

    explanation = form.get("explanation")
    uploaded_file = form.get("file")

    db_filename = None

    # Save file
    if uploaded_file and uploaded_file.filename:
        ext = os.path.splitext(uploaded_file.filename)[1]
        safe_name = f"{assignment_id}_{int(time.time())}{ext}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.file.read())

        db_filename = safe_name

    db = get_db()
    cursor = db.cursor()

    # Insert OR update submission
    cursor.execute("""
        INSERT INTO assignment_answers 
        (assignment_id, candidate_email, explanation, file_path, is_submitted)
        VALUES (%s, %s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE
            explanation = %s,
            file_path = COALESCE(%s, file_path),
            is_submitted = 1
    """, (assignment_id, candidate_email, explanation, db_filename,
          explanation, db_filename))

    db.commit()

    # Get submission_id for LLM thread
    cursor.execute("""
        SELECT id FROM assignment_answers
        WHERE assignment_id=%s AND candidate_email=%s
    """, (assignment_id, candidate_email))
    submission_id = cursor.fetchone()[0]

    # Update assignment status
    cursor.execute("""
        UPDATE assignments 
        SET verification_status='Processing'
        WHERE id=%s
    """, (assignment_id,))
    db.commit()

    cursor.close()
    db.close()

    # Start LLM thread (CORRECT)
    import threading
    threading.Thread(
        target=run_llm_verification_assignment,
        args=(submission_id,),       # <-- FIXED
        daemon=True
    ).start()

    return {"success": True}



@app.post("/assignment/llm-result")
def save_assignment_llm_result(
    assignment_id: int = Form(...),
    candidate_email: str = Form(...),
    score: int = Form(None),
    mismatch: int = Form(None),
    reason: str = Form(None)
):
    db = get_db()
    cursor = db.cursor()

    try:
        cursor.execute("""
            INSERT INTO llm_results (assignment_id, candidate_email, score, mismatch, reason)
            VALUES (%s, %s, %s, %s, %s)
        """, (assignment_id, candidate_email, score, mismatch, reason))

        db.commit()

        return {"success": True, "message": "LLM result saved"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        cursor.close()
        db.close()

@app.post("/test/llm-result")
async def save_test_llm_result(request: Request):
    """
    Accept both JSON and FormData safely.
    Store LLM results for TEST submissions.
    """

    # Try to read JSON first
    try:
        data = await request.json()
    except:
        # Fallback to FormData
        form = await request.form()
        data = dict(form)

    submission_id = data.get("submission_id")
    candidate_email = data.get("candidate_email")  # OPTIONAL NOW
    score = data.get("score")
    mismatch = data.get("mismatch")
    reason = data.get("reason")

    if not submission_id:
        return {"success": False, "error": "submission_id is required"}

    db = get_db()
    cursor = db.cursor()

    try:
        # ✔ CORRECT COLUMN NAME (assignment_id was wrong for tests)
        cursor.execute("""
            INSERT INTO llm_results (submission_id, candidate_email, score, mismatch, reason)
            VALUES (%s, %s, %s, %s, %s)
        """, (submission_id, candidate_email, score, mismatch, reason))

        db.commit()
        return {"success": True, "message": "LLM test result saved"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        cursor.close()
        db.close()


# Dynamic assignments table
@app.get("/load-dashboard", response_class=HTMLResponse)
def load_dashboard(request: Request):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # -----------------------------------------------
    # 1. LOAD ALL ASSIGNMENTS SENT TO CANDIDATES
    # -----------------------------------------------
    cursor.execute("""
        SELECT 
            ac.id AS candidate_row_id,
            ac.assignment_id,
            ac.candidate_name,
            ac.candidate_email,
            am.assignment_name AS title,
            am.due_date,
            'Assignment' AS job_type,

            aa.id AS submission_id,
            aa.is_submitted,
            aa.file_path,
            aa.explanation,

            lr.score AS llm_score,
            lr.mismatch AS mismatch,
            lr.reason AS llm_reason

        FROM assignment_candidates ac
        JOIN assignment_master am ON ac.assignment_id = am.id
        LEFT JOIN assignment_answers aa 
               ON aa.assignment_id = ac.assignment_id 
              AND aa.candidate_email = ac.candidate_email

        /* FIXED: LLM JOIN MUST USE SUBMISSION_ID */
        LEFT JOIN llm_results lr 
               ON lr.assignment_id = aa.id
              AND lr.candidate_email = ac.candidate_email

        ORDER BY ac.id DESC
    """)

    assignment_rows = cursor.fetchall()

    assignment_list = []
    for row in assignment_rows:
        assignment_list.append({
            "id": row["submission_id"] if row["submission_id"] else row["assignment_id"],
            "assignment_id": row["assignment_id"],
            "candidate_name": row["candidate_name"],
            "candidate_email": row["candidate_email"],
            "title": row["title"],
            "due_date": row["due_date"],
            "job_type": row["job_type"],

            "status": "Submitted" if row["is_submitted"] else "Pending",

            "verification_status": (
                "Verified" if row["llm_score"] is not None else "Pending"
            ),

            "llm_score": row["llm_score"],

            "file_or_link": f"/uploaded/{row['file_path']}"
                            if row["file_path"] else "-",

            "result_link": (
                f"/admin/view-assignment/{row['submission_id']}"
                if row["submission_id"] else "-"
            )
        })

    # -----------------------------------------------
    # 2. LOAD TEST SUBMISSIONS (NAME FIX APPLIED)
    # -----------------------------------------------
    cursor.execute("""
        SELECT 
            ts.id,
            ts.test_id,

            /* FIX NAME WITH FALLBACK */
            COALESCE(
                ts.candidate_name,
                (SELECT candidate_name 
                 FROM test_submissions 
                 WHERE test_id = ts.test_id
                   AND candidate_name IS NOT NULL
                   AND question_id IS NULL
                 ORDER BY id DESC LIMIT 1)
            ) AS candidate_name,

            /* FIX EMAIL WITH FALLBACK */
            COALESCE(
                ts.candidate_email,
                (SELECT candidate_email 
                 FROM test_submissions 
                 WHERE test_id = ts.test_id
                   AND candidate_email IS NOT NULL
                   AND question_id IS NULL
                 ORDER BY id DESC LIMIT 1)
            ) AS candidate_email,

            ts.verification_status,
            ts.llm_score,
            ts.created_at,

            t.test_name
        FROM test_submissions ts
        LEFT JOIN tests t ON ts.test_id = t.id
        WHERE ts.question_id IS NULL
        ORDER BY ts.created_at DESC
    """)

    test_rows = cursor.fetchall()

    merged_tests = []

    for t in test_rows:
        merged_tests.append({
            "id": t["id"],
            "candidate_name": t["candidate_name"],
            "candidate_email": t["candidate_email"],

            "title": t["test_name"] if t["test_name"] else "Test Submission",
            "job_type": "Test",
            "due_date": "-",

            "status": "Submitted",
            "verification_status": t["verification_status"],
            "llm_score": t["llm_score"],

            "file_or_link": f"/admin/view-assignment/{t['id']}",
            "result_link": f"/admin/view-assignment/{t['id']}",
        })

    cursor.close()
    db.close()

    combined = assignment_list + merged_tests

    return templates.TemplateResponse(
        "dashboard_content.html",
        {"request": request, "assignments": combined}
    )


# Dynamic create assignment form
@app.get("/load-create-assignment", response_class=HTMLResponse)
def load_create_assignment(request: Request):
    return templates.TemplateResponse("create_assignment_content.html", {"request": request, "error": "", "success": ""})

@app.get("/load-tests", response_class=HTMLResponse)
def load_tests(request: Request):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM tests ORDER BY created_at DESC")
    tests = cursor.fetchall()

    cursor.close()
    db.close()

    return templates.TemplateResponse(
        "tests_list.html",
        {"request": request, "tests": tests}
    )

@app.get("/admin/test/{test_id}", response_class=HTMLResponse)
def view_test_detail(request: Request, test_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Fetch test information
    cursor.execute("""
        SELECT * FROM tests WHERE id = %s
    """, (test_id,))
    test = cursor.fetchone()

    if not test:
        return HTMLResponse("<h3>Test not found</h3>")

    # Fetch test questions with full question text
    cursor.execute("""
        SELECT tq.question_marks, q.question_text
        FROM test_questions tq
        JOIN questions q ON q.id = tq.question_id
        WHERE tq.test_id = %s
    """, (test_id,))
    questions = cursor.fetchall()

    cursor.close()
    db.close()

    return templates.TemplateResponse(
        "test_detail.html",
        {
            "request": request,
            "test": test,
            "questions": questions
        }
    )


@app.get("/admin/question-bank", response_class=HTMLResponse)
def question_bank_page(request: Request):

    admin_email = request.cookies.get("admin_email")
    if not admin_email:
        return RedirectResponse("/login")

    db = get_db()
    cursor = db.cursor(dictionary=True)

    # ---------------------------------------------
    # Load all questions
    # ---------------------------------------------
    cursor.execute("SELECT * FROM questions ORDER BY created_at DESC")
    rows = cursor.fetchall()

    # ---------------------------------------------
    # Decode mcq_options JSON safely
    # ---------------------------------------------
    import json
    for r in rows:
        if r.get("mcq_options"):
            try:
                r["options"] = json.loads(r["mcq_options"])
            except:
                r["options"] = {}
        else:
            r["options"] = {}

    # ---------------------------------------------
    # Load DISTINCT Skill Tags
    # ---------------------------------------------
    cursor.execute("SELECT DISTINCT skill_tag FROM questions ORDER BY skill_tag ASC")
    skills = [row["skill_tag"] for row in cursor.fetchall()]

    # ---------------------------------------------
    # Load DISTINCT Experience
    # ---------------------------------------------
    cursor.execute("SELECT DISTINCT experience_tag FROM questions ORDER BY experience_tag ASC")
    experiences = [row["experience_tag"] for row in cursor.fetchall()]

    cursor.close()
    db.close()

    # ---------------------------------------------
    # Return Template
    # ---------------------------------------------
    return templates.TemplateResponse(
        "question_bank.html",
        {
            "request": request,
            "questions": rows,
            "skills": skills,
            "experiences": experiences,
            "admin_email": admin_email
        }
    )

# --------------------------------------------------------------
# SAFE JSON EXTRACTOR (Fixes QWEN extra text / bad formatting)
# --------------------------------------------------------------
def extract_json_safe(text):
    import json, re

    # Normalize text
    cleaned = text.replace("```json", "").replace("```", "").strip()

    # Find ALL possible JSON blocks
    candidates = re.findall(r"\{[\s\S]*?\}", cleaned)

    if not candidates:
        return None

    # Try loading each JSON block until a valid one is found
    for block in candidates:
        try:
            return json.loads(block)
        except:
            # Try auto-fixing common trailing commas
            fixed = re.sub(r",\s*([}\]])", r"\1", block)
            try:
                return json.loads(fixed)
            except:
                continue

    return None


# ==============================================================
# ==================== JSON CLEANER =============================
# ==============================================================

def extract_clean_json(text):
    """Extract first valid {...} JSON block from messy AI output."""
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)

    # Fix common issues
    json_str = json_str.replace("“", "\"").replace("”", "\"")
    json_str = json_str.replace("‘", "'").replace("’", "'")
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    try:
        return json.loads(json_str)
    except:
        return None


def extract_json_safe(text):
    """
    Extract first valid JSON object from messy AI output.
    Works without recursive regex.
    """

    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    end = None

    # Manual parsing (safe & stable)
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if end is None:
        return None

    json_str = text[start:end].strip()

    # Try decode
    try:
        data = json.loads(json_str)
        # Must contain essential keys
        if all(k in data for k in ["A", "B", "C", "D", "correct"]):
            return data
        return None
    except:
        return None

# ---------------------- GLOBAL MCQ PROMPT ----------------------
def build_mcq_prompt(question_text):
    return f"""
You MUST output ONLY a VALID JSON object.

NO markdown.
NO explanation.
NO extra sentences.
NO backticks.

Output MUST start with '{{' and end with '}}'.

JSON structure MUST be EXACTLY:

{{
  "A": "option text A",
  "B": "option text B",
  "C": "option text C",
  "D": "option text D",
  "correct": "A"
}}

RULES:
- Provide ONLY the JSON above — nothing else.
- Each option MUST be a short incorrect or correct answer.
- EXACTLY ONE option must be correct.
- "correct" MUST be: "A", "B", "C" OR "D" ONLY.

Now generate the MCQ for this question:

{question_text}
"""


# ---------------------- GLOBAL QWEN CALL ----------------------
def generate_answer_with_qwen(prompt):
    try:
        out = llm.generate(prompt, max_tokens=100, temp=0.1)
        return out.strip()
    except Exception as e:
        print("QWEN ERROR:", e)
        return None

# ==============================================================
# ==================== AI GENERATE QUESTION =====================
# ==============================================================

@app.post("/admin/ai-generate-question")
def ai_generate_question(
    request: Request,
    question_text: str = Form(...),
    question_type: str = Form(...),
    skill_tag: str = Form(...),
    experience_tag: str = Form(...)
):

    if question_type == "mcq":
        prompt = build_mcq_prompt(question_text)
        raw = generate_answer_with_qwen(prompt)

        if not raw:
            return JSONResponse({"success": False, "error": "AI returned empty"})

        data = extract_json_safe(raw)
        if not data:
            return JSONResponse({"success": False, "error": "Invalid JSON from AI"})

        return JSONResponse({
            "success": True,
            "A": data["A"],
            "B": data["B"],
            "C": data["C"],
            "D": data["D"],
            "correct": data["correct"]
        })

    # TEXT TYPE
    prompt = f"Write a 2–3 line short answer:\n{question_text}"
    ans = generate_answer_with_qwen(prompt)

    if not ans:
        return JSONResponse({"success": False, "error": "AI failed"})

    return JSONResponse({"success": True, "answer": ans})


# ==============================================================
# ========================= ADD QUESTION ========================
# ==============================================================

@app.post("/admin/add-question")
def add_question(
    request: Request,
    question_text: str = Form(...),
    question_type: str = Form(...),
    skill_tag: str = Form(...),
    experience_tag: str = Form(...),
    option_a: str = Form(None),
    option_b: str = Form(None),
    option_c: str = Form(None),
    option_d: str = Form(None),
    correct_option: str = Form(None),
    model_answer: str = Form(None)
):

    admin_email = request.cookies.get("admin_email")
    if not admin_email:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    question_text = question_text.strip()

    mcq_options_json = None
    correct_opt = None
    model_ans = None

    # ======================================================
    #                MCQ AUTO GENERATION
    # ======================================================
    if question_type == "mcq":

        # If any field is missing → auto-generate
        if not (option_a and option_b and option_c and option_d and correct_option):

            prompt = build_mcq_prompt(question_text)
            raw = generate_answer_with_qwen(prompt)

            if not raw:
                return JSONResponse({"error": "AI generation failed"}, status_code=500)

            # Extract JSON block
            data = extract_json_safe(raw)
            if not data:
                return JSONResponse({"error": "AI returned invalid JSON"}, status_code=500)

            # Assign values
            option_a = data["A"]
            option_b = data["B"]
            option_c = data["C"]
            option_d = data["D"]
            correct_opt = data["correct"].strip().upper()

        else:
            correct_opt = correct_option.strip().upper()

        # Build JSON for DB
        mcq_options_json = json.dumps({
            "A": option_a.strip(),
            "B": option_b.strip(),
            "C": option_c.strip(),
            "D": option_d.strip(),
        })

    # ======================================================
    #                TEXT ANSWER GENERATION
    # ======================================================
    else:
        if model_answer:
            model_ans = model_answer.strip()
        else:
            prompt = f"Write a short 2–3 line answer:\n{question_text}"
            model_ans = generate_answer_with_qwen(prompt)

    # ======================================================
    #                SAVE TO DATABASE
    # ======================================================
    db = get_db()
    cursor = db.cursor()

    query = """
        INSERT INTO questions 
        (question_text, question_type, mcq_options, correct_option, model_answer,
         skill_tag, experience_tag, created_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        question_text,
        question_type,
        mcq_options_json,
        correct_opt,
        model_ans,
        skill_tag,
        experience_tag,
        admin_email
    )

    cursor.execute(query, values)
    db.commit()
    cursor.close()
    db.close()

    return JSONResponse({"success": True})


@app.get("/admin/create-test", response_class=HTMLResponse)
def create_test_page(request: Request, ids: str):
    # Clean split
    id_list = [x for x in ids.split(",") if x.strip()]

    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Fetch questions
    query = "SELECT * FROM questions WHERE id IN (" + ",".join(["%s"] * len(id_list)) + ")"
    cursor.execute(query, id_list)
    questions = cursor.fetchall()

    cursor.close()
    db.close()

    # FIX: return ids
    return templates.TemplateResponse(
        "create_test.html",
        {
            "request": request,
            "questions": questions,
            "ids": ids   # ← REQUIRED FIX
        }
    )


@app.post("/admin/save-test", response_class=HTMLResponse)
async def save_test(request: Request):
    form = await request.form()

    test_name = form.get("test_name")
    total_marks = form.get("total_marks")
    duration = form.get("duration")
    question_ids_raw = form.get("question_ids")

    # Validation
    if not total_marks or not duration or not question_ids_raw:
        return HTMLResponse("<h3>Error: Missing values</h3>")

    total_marks = int(total_marks)
    duration = int(duration)

    question_ids = [int(q) for q in question_ids_raw.split(",") if q.strip()]

    if not question_ids:
        return HTMLResponse("<h3>Error: No questions selected.</h3>")

    db = get_db()
    cursor = db.cursor()

    # Insert test
    cursor.execute("""
        INSERT INTO tests (test_name, total_marks, duration_minutes)
        VALUES (%s, %s, %s)
    """, (test_name, total_marks, duration))

    test_id = cursor.lastrowid

    # Mark distribution
    per_question = total_marks // len(question_ids)

    # Insert questions into test
    for qid in question_ids:
        cursor.execute("""
            INSERT INTO test_questions (test_id, question_id, question_marks)
            VALUES (%s, %s, %s)
        """, (test_id, qid, per_question))

    db.commit()
    cursor.close()
    db.close()

    # 🎉 Return success HTML INSIDE same page
    return templates.TemplateResponse(
        "create_test.html",
        {
            "request": request,
            "success": True,
            "message": f"Test Created Successfully! (ID: {test_id})",
            "test_id": test_id,
            "questions": []  # not needed now
        }
    )

fm = FastMail(conf)

async def send_email(to_email, subject, message):
    msg = MessageSchema(
        subject=subject,
        recipients=[to_email],   # list required
        body=message,
        subtype="plain"
    )

    try:
        await fm.send_message(msg)
        print("Email sent to:", to_email)
        return True
    except Exception as e:
        print("Email Error:", e)
        return False

@app.post("/admin/send-test")
async def send_test(request: Request):
    form = await request.form()
    test_id = form.get("test_id")

    if not test_id:
        return {"success": False, "error": "Missing test_id"}

    candidates = []

    for i in range(1, 21):
        name = form.get(f"name{i}")
        email = form.get(f"email{i}")

        if email and email.strip():
            candidates.append({
                "name": name.strip() if name else "",
                "email": email.strip()
            })

    sent_to = []

    for c in candidates:
        candidate_email = c["email"]
        candidate_name = c["name"] if c["name"] else "Candidate"

        # temp password
        temp_password = secrets.token_hex(3)

        # Create magic login link (NO assignment)
        test_link = f"http://localhost:8000/candidate-login?test_id={test_id}&email={candidate_email}&temp_password={temp_password}"

        subject = "Your Online Assessment is Ready"

        message = f"""
Hello {candidate_name},

You have been invited to take your online assessment.

Click the link below to begin:
{test_link}

Best regards,
Recruitment Team
"""

        sent_to.append(candidate_email)
        await send_email(candidate_email, subject, message)

    return {
        "success": True,
        "sent_to": sent_to
    }


@app.get("/test/start/{test_id}", response_class=HTMLResponse)
async def start_test(request: Request, test_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Load test
    cursor.execute("SELECT * FROM tests WHERE id=%s", (test_id,))
    test = cursor.fetchone()

    # Load questions count for instructions page
    cursor.execute("""
        SELECT q.id, q.question_text
        FROM test_questions tq
        JOIN questions q ON q.id = tq.question_id
        WHERE tq.test_id=%s
    """, (test_id,))
    questions = cursor.fetchall()

    cursor.close()
    db.close()

    if not test:
        return HTMLResponse("<h3>Invalid Test Link</h3>")

    return templates.TemplateResponse("test_instructions.html", {
        "request": request,
        "test": test,
        "questions": questions
    })


@app.post("/test/start-session")
async def start_test_session(request: Request):
    form = await request.form()

    test_id = form.get("test_id")
    candidate_name = form.get("candidate_name")
    candidate_email = form.get("candidate_email")

    if not test_id:
        return JSONResponse({"success": False, "error": "Missing test_id"}, status_code=400)

    db = get_db()
    cursor = db.cursor()

    try:
        cursor.execute("""
            INSERT INTO test_submissions
            (test_id, candidate_name, candidate_email, question_id, fullscreen_attempts, is_flagged, created_at)
            VALUES (%s, %s, %s, NULL, 0, 0, %s)
        """, (test_id, candidate_name, candidate_email, datetime.utcnow()))

        submission_id = cursor.lastrowid
        db.commit()

    except Exception as e:
        db.rollback()
        cursor.close()
        db.close()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    cursor.close()
    db.close()

    return JSONResponse({"success": True, "submission_id": submission_id})


@app.get("/test/fullscreen/{test_id}", response_class=HTMLResponse)
async def fullscreen_test(request: Request, test_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Fetch test
    cursor.execute("SELECT * FROM tests WHERE id=%s", (test_id,))
    test = cursor.fetchone()

    # Fetch mapped questions
    cursor.execute("""
        SELECT q.id, q.question_text, q.question_type, q.mcq_options
        FROM test_questions tq
        JOIN questions q ON tq.question_id = q.id
        WHERE tq.test_id = %s
    """, (test_id,))
    questions = cursor.fetchall()

    cursor.close()
    db.close()

    if not test:
        return HTMLResponse("<h3>Invalid Test</h3>")

    import json
    for q in questions:
        if q["question_type"] == "mcq" and q["mcq_options"]:
            opts = json.loads(q["mcq_options"])
            q["mcq_options"] = [f"{k}: {v}" for k, v in opts.items()]
        else:
            q["mcq_options"] = []

    # ⭐ Extract submission_id from URL
    sid = request.query_params.get("sid", "")

    # ⭐ FIX — read candidate info from cookie
    candidate_email = request.cookies.get("candidate_email", "")
    candidate_name = request.cookies.get("candidate_name", "")

    return templates.TemplateResponse("test_fullscreen.html", {
        "request": request,
        "test": test,
        "questions": questions,
        "sid": sid,
        "candidate_email": candidate_email,   # ✔ added
        "candidate_name": candidate_name      # ✔ added
    })


@app.post("/test/submit")
async def submit_test(request: Request):
    form = await request.form()

    test_id = form.get("test_id")
    candidate_name = form.get("candidate_name")
    candidate_email = form.get("candidate_email")
    submission_id = form.get("submission_id")   # ⭐ REQUIRED

    # -----------------------------------------
    # Extract all MCQ answers
    # -----------------------------------------
    answers = {k: v for k, v in form.items() if k.startswith("answer_")}

    # Convert answers into event_log
    event_log = ""
    for key, ans in answers.items():
        q_id = key.replace("answer_", "")
        event_log += f"Q{q_id}: {ans}\n"

    db = get_db()
    cursor = db.cursor()

    # -----------------------------------------
    # ⭐ UPDATE the main submission row
    # -----------------------------------------
    cursor.execute("""
        UPDATE test_submissions
        SET event_log = %s,
            candidate_name = %s,
            candidate_email = %s,
            verification_status = 'Pending'
        WHERE id = %s
    """, (event_log, candidate_name, candidate_email, submission_id))

    # -----------------------------------------
    # Insert each answer as additional rows
    # -----------------------------------------
    for key, answer in answers.items():
        q_id = key.replace("answer_", "")
        cursor.execute("""
            INSERT INTO test_submissions (test_id, question_id, answer)
            VALUES (%s, %s, %s)
        """, (test_id, q_id, answer))

    # -----------------------------------------
    # Mark main row as PROCESSING
    # -----------------------------------------
    cursor.execute("""
        UPDATE test_submissions
        SET verification_status = 'Processing'
        WHERE id = %s
    """, (submission_id,))
    
    db.commit()
    cursor.close()
    db.close()

    # -----------------------------------------
    # Start Background LLM Verification
    # -----------------------------------------
    import threading
    threading.Thread(
        target=run_llm_verification_test,
        args=(submission_id, test_id, candidate_email),
        daemon=True
    ).start()

    return {
        "success": True,
        "submission_id": submission_id
    }

    
@app.post("/test/violation")
async def test_violation(data: dict = Body(...)):
    submission_id = data.get("submission_id")

    if not submission_id:
        return {"success": False, "error": "Missing submission_id"}

    db = get_db()
    cursor = db.cursor()

    # Immediately terminate test
    cursor.execute("""
        UPDATE test_submissions
        SET verification_status='Terminated',
            is_flagged=1
        WHERE id=%s
    """, (submission_id,))

    db.commit()
    cursor.close()
    db.close()

    return {
        "success": True,
        "action": "end_test"
    }

@app.get("/test-ended", response_class=HTMLResponse)
def test_ended(request: Request):
    return HTMLResponse("""
        <h2>Your test has been terminated.</h2>
        <p>You exited fullscreen or ended the test manually.</p>
    """)

@app.get("/admin/view-assignment/{submission_id}")
async def admin_view_assignment(request: Request, submission_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Load assignment submission
    cursor.execute("""
        SELECT 
            aa.id AS submission_id,
            aa.assignment_id,
            ac.candidate_name,
            aa.candidate_email,
            am.assignment_name AS title,
            am.scenario_text AS description,
            am.due_date,
            'Assignment' AS job_type,
            aa.file_path,
            aa.explanation,
            aa.is_submitted,
            aa.manual_score
        FROM assignment_answers aa
        JOIN assignment_candidates ac 
            ON ac.assignment_id = aa.assignment_id
           AND ac.candidate_email = aa.candidate_email
        JOIN assignment_master am 
            ON am.id = aa.assignment_id
        WHERE aa.id=%s
    """, (submission_id,))

    submission = cursor.fetchone()

    # ----------------- TEST SUBMISSION -----------------
    if not submission:

        cursor.execute("""
            SELECT 
                ts.id AS submission_id,
                ts.test_id AS assignment_id,
                ts.candidate_name,
                ts.candidate_email,

                t.test_name AS title,
                NULL AS description,       -- FIXED (your DB has no description)
                t.duration_minutes AS due_date,

                'Test' AS job_type,

                NULL AS file_path,
                NULL AS explanation,
                1 AS is_submitted,

                ts.verification_status,
                ts.llm_score,
                ts.llm_feedback AS llm_result,
                ts.mcq_score,
                ts.mcq_total

            FROM test_submissions ts
            LEFT JOIN tests t ON ts.test_id = t.id
            WHERE ts.id=%s AND ts.question_id IS NULL
        """, (submission_id,))

        test = cursor.fetchone()

        cursor.close()
        db.close()

        if not test:
            return HTMLResponse("<h3>Submission not found.</h3>", status_code=404)

        test["submission_type"] = "test"

        return templates.TemplateResponse("admin_view_assignment.html", {
            "request": request,
            "submission": test,
            "description_items": [],
            "mcqs": [],
            "mcq_score": test.get("mcq_score", 0),
            "mcq_total": test.get("mcq_total", 0),
            "attachments": []
        })

    # ----------------- ASSIGNMENT SUBMISSION -----------------
    submission["submission_type"] = "assignment"

    description_items = []
    try:
        parsed = json.loads(submission["description"])
        if isinstance(parsed, list):
            description_items = parsed
    except:
        pass

    cursor.execute("""
        SELECT score, mismatch, reason
        FROM llm_results
        WHERE assignment_id=%s AND candidate_email=%s
        ORDER BY created_at DESC LIMIT 1
    """, (submission["submission_id"], submission["candidate_email"]))

    llm = cursor.fetchone()
    submission["llm_score"] = llm["score"] if llm else None
    submission["llm_feedback"] = llm["reason"] if llm else None
    submission["verification_status"] = "Verified" if llm else "Pending"

    # Attachments
    attachments = []

    def normalize_path(path):
        path = path.lstrip("/")
        if path.startswith("uploads/"):
            path = path.replace("uploads/", "")
        return f"/uploads/{path}"

    if submission.get("file_path"):
        file_name = submission["file_path"].split("/")[-1]
        attachments.append({
            "file_name": file_name,
            "file_path": normalize_path(submission["file_path"])
        })

    cursor.execute("""
        SELECT file_name, file_path 
        FROM assignment_attachments 
        WHERE scenario_id=%s
    """, (submission["assignment_id"],))

    for f in cursor.fetchall():
        attachments.append({
            "file_name": f["file_name"],
            "file_path": normalize_path(f["file_path"])
        })

    cursor.close()
    db.close()

    return templates.TemplateResponse("admin_view_assignment.html", {
        "request": request,
        "submission": submission,
        "description_items": description_items,
        "mcqs": [],
        "mcq_score": 0,
        "mcq_total": 0,
        "attachments": attachments
    })


@app.post("/admin/manual-score/{submission_id}")
async def save_manual_score(request: Request, submission_id: int, manual_score: int = Form(...)):
    db = get_db()
    cursor = db.cursor()

    try:
        cursor.execute("""
            UPDATE assignment_answers
            SET manual_score = %s
            WHERE id = %s
        """, (manual_score, submission_id))

        db.commit()

    except Exception as e:
        cursor.close()
        db.close()
        return HTMLResponse(f"<h3>Error saving score: {e}</h3>")

    cursor.close()
    db.close()

    return RedirectResponse(url=f"/admin/view-assignment/{submission_id}", status_code=302)


import re

ALLOWED_EXT = {
    ".cs", ".py", ".js", ".ts", ".html", ".css",
    ".sql", ".json", ".xml", ".md", ".txt", ".csv"
}

IGNORE_FOLDERS = {"node_modules", "dist", "build", "vendor", "bootstrap"}
MAX_FILES = 20
MAX_CHARS = 1200


def extract_useful_content(zip_path):
    extract_dir = tempfile.mkdtemp()
    useful_files = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    except:
        return "Zip not readable."

    for root, dirs, files in os.walk(extract_dir):
        if any(x.lower() in root.lower() for x in IGNORE_FOLDERS):
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXT:
                useful_files.append(os.path.join(root, file))

    useful_files = useful_files[:MAX_FILES]

    summary = "=== Filtered Project Files ===\n"
    for file_path in useful_files:
        file_name = os.path.basename(file_path)
        summary += f"\n--- FILE: {file_name} ---\n"
        try:
            with open(file_path, "r", errors="ignore") as f:
                summary += f.read()[:MAX_CHARS] + "\n"
        except:
            summary += "(Unreadable file)\n"

    return summary


# ============================================================
# 🚀 GPT4ALL OFFLINE MODEL LOADER
# ============================================================

def load_offline_llm():
    MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    MODEL_PATH = "D:/Application_form/GPT4ALL/models"
    return GPT4All(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        allow_download=False,
        verbose=False
    )


# ============================================================
# 🚀 EXTRACT SCORE + ELIGIBILITY FROM LLM OUTPUT
# ============================================================

def extract_score_and_eligibility(text):
    try:
        lines = text.split("\n")
        score_line = next((x for x in lines if "Score:" in x), None)

        if score_line:
            score_str = "".join([c for c in score_line if c.isdigit()])
            score = int(score_str) if score_str.isdigit() else 0
        else:
            score = 0

        eligible = "Eligible" if score >= 5 else "Not Eligible"

        return score, eligible
    except:
        return 0, "Not Eligible"


# ============================================================
# 🚀 ASSIGNMENT LLM VERIFICATION
# ============================================================

def run_llm_verification_assignment(submission_id):
    print("LLM: Assignment verification started...")

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            aa.id AS submission_id,
            aa.assignment_id,
            aa.candidate_email,
            aa.file_path,
            aa.explanation,
            am.assignment_name,
            am.scenario_text
        FROM assignment_answers aa
        JOIN assignment_master am
            ON am.id = aa.assignment_id
        WHERE aa.id=%s
    """, (submission_id,))

    submission = cursor.fetchone()

    if not submission:
        print("LLM ERROR: Submission not found")
        cursor.close()
        db.close()
        return

    assignment_id = submission["assignment_id"]
    candidate_email = submission["candidate_email"]
    file_path = submission.get("file_path")
    explanation = submission.get("explanation")

    cursor.close()
    db.close()


    # --------------------------
    # Build submission summary
    # --------------------------
    project_summary = "=== Candidate Submission ===\n"

    if file_path and os.path.exists(file_path):
        if zipfile.is_zipfile(file_path):
            project_summary += extract_useful_content(file_path)
        else:
            try:
                with open(file_path, "r", errors="ignore") as f:
                    project_summary += f.read()[:1500]
            except:
                project_summary += "(File unreadable)"
    else:
        project_summary += explanation or "(No explanation text submitted)"


    # --------------------------
    # Build LLM prompt
    # --------------------------
    prompt = f"""
Strict assignment evaluator.

Candidate: {candidate_email}
Submission ID: {submission_id}
Assignment ID: {assignment_id}

PROJECT DATA:
{project_summary}

RULES:
• Score must be 0–10 only
• Score >= 5 = Eligible
• Provide detailed explanation
"""


    # --------------------------
    # Run Local LLM
    # --------------------------
    llm = load_offline_llm()
    llm_output = llm.generate(prompt)

    score, eligible = extract_score_and_eligibility(llm_output)


    # ==================================================
    # FIXED — SAVE USING submission_id (NOT assignment_id)
    # ==================================================
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            INSERT INTO llm_results (assignment_id, candidate_email, score, mismatch, reason)
            VALUES (%s, %s, %s, %s, %s)
        """, (submission_id, candidate_email, score, 0, llm_output))

        db.commit()
    except Exception as e:
        print("LLM SAVE ERROR:", e)
    finally:
        cursor.close()
        db.close()


    # --------------------------
    # Update assignment_answers
    # --------------------------
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            UPDATE assignment_answers
            SET 
                llm_score=%s,
                llm_feedback=%s,
                verification_status='Verified'
            WHERE id=%s
        """, (score, llm_output, submission_id))

        db.commit()
    except Exception as e:
        print("LLM UPDATE ERROR:", e)
    finally:
        cursor.close()
        db.close()

    print("LLM Assignment verification completed.")

# ============================================================
# 🚀 TEST LLM VERIFICATION
# ============================================================

def run_llm_verification_test(submission_id, test_id, candidate_email):
    print("LLM: Test verification started...")

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT ts.*, t.total_marks, t.test_name
        FROM test_submissions ts
        LEFT JOIN tests t ON ts.test_id = t.id
        WHERE ts.id=%s
    """, (submission_id,))
    sub = cursor.fetchone()

    if not sub:
        print("LLM ERROR: Test submission not found")
        cursor.close()
        db.close()
        return

    cursor.close()
    db.close()

    # Build Prompt
    prompt = f"""
Strict test evaluator.

TEST NAME: {sub.get("test_name")}
Candidate Email: {candidate_email}

ANSWERS:
{sub.get("event_log")}

RULES:
• Evaluate strictly.
• Score out of {sub.get("total_marks", 10)}
• >=50% = Eligible.
"""

    # Run LLM
    llm = load_offline_llm()
    output = llm.generate(prompt)
    score, eligibility = extract_score_and_eligibility(output)

    # Save LLM results to backend (✔ FIX: submission_id used)
    try:
        requests.post("http://localhost:8000/test/llm-result", data={
            "submission_id": submission_id, 
            "test_id": test_id,
            "candidate_email": candidate_email,
            "score": score,
            "mismatch": 0,
            "reason": output
        })
    except Exception as e:
        print("LLM RESULT SAVE ERROR:", e)

    # Update main submission table
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        UPDATE test_submissions
        SET verification_status='Verified',
            llm_score=%s,
            llm_feedback=%s
        WHERE id=%s
    """, (score, output, submission_id))
    db.commit()
    cursor.close()
    db.close()

    print("LLM Test verification completed.")


  
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
