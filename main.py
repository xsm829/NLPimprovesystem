from fastapi import FastAPI, Request, Response, Form, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import generate_password_hash, check_password_hash
import os
import random
import string
from typing import Optional, List
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from wordlearn import translate_text, translate_sentences, process_word_learning
import edge_tts
import asyncio
import re
from collections import Counter
from textblob import TextBlob
from fastapi.staticfiles import StaticFiles
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# FastAPI app
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="abyu7689fghjklzxcvbnm")

# 配置模板
templates = Jinja2Templates(directory="templates")

# 配置静态文件 - 修改为先挂载静态文件
app.mount(path="/static", app=StaticFiles(directory="./static"), name="static")
# 配置静态文件路径
# 数据库配置
SQLALCHEMY_DATABASE_URL = "mysql://root:xsm829@localhost/nlp_wordslearning"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 邮件配置
mail_config = ConnectionConfig(
    MAIL_USERNAME="xieshumeng829@163.com",
    MAIL_PASSWORD="JNxV33AxUPQbsf8a",
    MAIL_FROM="xieshumeng829@163.com",
    MAIL_PORT=465,
    MAIL_SERVER="smtp.163.com",
    MAIL_SSL_TLS=True,
    USE_CREDENTIALS=True,
    MAIL_STARTTLS=False
)

# 文件上传配置
UPLOAD_FOLDER = os.path.join("static", "avatars")  # 修改为正确的相对路径
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 用户模型
class User(Base):
    __tablename__ = "user"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password = Column(String(500), nullable=False)
    reset_token = Column(String(100), unique=True)
    avatar = Column(String(200))
    phone = Column(String(20))
    address = Column(String(200))

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 路由
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    if "username" not in request.session:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/register")
@app.post("/register")
async def register(request: Request, db: Session = Depends(get_db)):
    if request.method == "POST":
        form = await request.form()
        username = form.get("username")
        email = form.get("email")
        password = form.get("password")

        if db.query(User).filter(User.username == username).first():
            if "_flash_messages" not in request.session:
                request.session["_flash_messages"] = []
            request.session["_flash_messages"].append({"message": "用户名已存在", "category": "error"})
            return templates.TemplateResponse("register.html", {"request": request, "messages": get_flashed_messages(request)})

        if db.query(User).filter(User.email == email).first():
            if "_flash_messages" not in request.session:
                request.session["_flash_messages"] = []
            request.session["_flash_messages"].append({"message": "邮箱已被注册", "category": "error"})
            return templates.TemplateResponse("register.html", {"request": request, "messages": get_flashed_messages(request)})

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.add(new_user)
        db.commit()

        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse("register.html", {"request": request, "messages": get_flashed_messages(request)})

def get_flashed_messages(request: Request):
    flash_messages = []
    if "_flash_messages" in request.session:
        flash_messages = request.session["_flash_messages"]
        del request.session["_flash_messages"]
    return flash_messages

@app.get("/login")
@app.post("/login")
async def login(request: Request, db: Session = Depends(get_db)):
    if request.method == "POST":
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        user = db.query(User).filter(User.username == username).first()
        if user and check_password_hash(user.password, password):
            request.session["username"] = username
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        else:
            if "_flash_messages" not in request.session:
                request.session["_flash_messages"] = []
            request.session["_flash_messages"].append({"message": "用户名或密码错误", "category": "error"})
            return templates.TemplateResponse("login.html", {"request": request, "messages": get_flashed_messages(request)})

    return templates.TemplateResponse("login.html", {"request": request, "messages": get_flashed_messages(request)})

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("username", None)
    return RedirectResponse(url="/login")

@app.get("/personal_information")
@app.post("/personal_information")
async def personal_information(request: Request, db: Session = Depends(get_db)):
    if "username" not in request.session:
        return RedirectResponse(url="/login")

    user = db.query(User).filter(User.username == request.session["username"]).first()

    if request.method == "POST":
        form = await request.form()
        user.username = form.get("username", user.username)
        user.email = form.get("email", user.email)

        old_password = form.get("password")
        new_password = form.get("new_password")
        confirm_password = form.get("confirm_password")

        if old_password and not check_password_hash(user.password, old_password):
            raise HTTPException(status_code=400, detail="原密码不正确！")

        if new_password:
            if new_password != confirm_password:
                raise HTTPException(status_code=400, detail="新密码和确认密码不一致！")
            user.password = generate_password_hash(new_password)

        db.commit()
        return RedirectResponse(url="/personal_information", status_code=303)

    # 修改头像文件路径处理
    avatar_filename = user.avatar if user.avatar else 'bg2.jpg'
    return templates.TemplateResponse(
        "personal_information.html", 
        {"request": request, "user": user, "avatar_filename": avatar_filename}
    )

@app.post("/upload_avatar")
async def upload_avatar(
    request: Request,
    avatar: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if "username" not in request.session:
        return RedirectResponse(url="/login")

    user = db.query(User).filter(User.username == request.session["username"]).first()

    if avatar.filename != "":
        # 保留原始文件名
        filename = secure_filename(avatar.filename)  # 使用 secure_filename 清理文件名
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # 检查文件名是否已存在，如果存在则添加随机数
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            random_suffix = random.randint(1000, 9999)
            filename = f"{name}_{random_suffix}{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(file_path, "wb") as buffer:
            content = await avatar.read()
            buffer.write(content)

        # 删除旧头像文件
        if user.avatar and user.avatar != "bg2.jpg":
            old_file = os.path.join(UPLOAD_FOLDER, user.avatar)
            if os.path.exists(old_file):
                os.remove(old_file)

        # 更新用户头像
        user.avatar = filename
        db.commit()

    return RedirectResponse(url="/personal_information", status_code=303)

@app.post("/update_profile")
async def update_profile(request: Request, db: Session = Depends(get_db)):
    if "username" not in request.session:
        return RedirectResponse(url="/login")
    
    form = await request.form()
    user = db.query(User).filter(User.username == request.session["username"]).first()

    user.email = form.get("email", "")
    user.address = form.get("address", "")

    if form.get("password"):
        user.password = generate_password_hash(form.get("password"))
    
    db.commit()
    return RedirectResponse(url="/personal_information", status_code=303)

@app.get("/forgot_password")
@app.post("/forgot_password")
async def forgot_password(request: Request, db: Session = Depends(get_db)):
    if request.method == "POST":
        form = await request.form()
        email = form.get("email")
        user = db.query(User).filter(User.email == email).first()

        if user:
            token = "".join(random.choices(string.ascii_letters + string.digits, k=50))
            user.reset_token = token
            db.commit()

            # 发送重置密码邮件
            message = MessageSchema(
                subject="密码重置请求",
                recipients=[email],
                body=f"点击以下链接重置密码：http://localhost:8000/reset_password/{token}",
                subtype="plain"  # 添加 subtype 字段
            )

            fm = FastMail(mail_config)
            await fm.send_message(message)

            return RedirectResponse(url="/login", status_code=303)
        else:
            if "_flash_messages" not in request.session:
                request.session["_flash_messages"] = []
            request.session["_flash_messages"].append({"message": "该邮箱未注册", "category": "error"})
            return templates.TemplateResponse("forgot_password.html", {"request": request, "messages": get_flashed_messages(request)})

    return templates.TemplateResponse("forgot_password.html", {"request": request, "messages": get_flashed_messages(request)})

@app.get("/reset_password/{token}")
@app.post("/reset_password/{token}")
async def reset_password(token: str, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.reset_token == token).first()

    if not user:
        if "_flash_messages" not in request.session:
            request.session["_flash_messages"] = []
        request.session["_flash_messages"].append({"message": "无效的重置链接", "category": "error"})
        return templates.TemplateResponse("reset_password.html", {"request": request, "messages": get_flashed_messages(request)})

    if request.method == "POST":
        form = await request.form()
        password = form.get("password")
        user.password = generate_password_hash(password)
        user.reset_token = None
        db.commit()

        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse("reset_password.html", {"request": request, "messages": get_flashed_messages(request)})

# 单词学习相关路由
@app.get("/word_learning", response_class=HTMLResponse)
async def word_learning_page(request: Request):
    """渲染单词学习页面"""
    if "username" not in request.session:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("word_learning.html", {"request": request})

@app.post("/learn_word")
async def learn_word(request: Request, word: str = Form(...)):
    """处理单词学习请求"""
    try:
        result = await process_word_learning(word)
        return templates.TemplateResponse("word_learning.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse("word_learning.html", {"request": request, "error": str(e)})

@app.post("/translate")
async def translate(request: Request, text: str = Form(...), from_lang: str = Form("en"), to_lang: str = Form("zh")):
    """文本翻译"""
    try:
        translated_text = translate_text(text, from_lang, to_lang)
        return templates.TemplateResponse("word_learning.html", {"request": request, "translated_text": translated_text})
    except Exception as e:
        return templates.TemplateResponse("word_learning.html", {"request": request, "error": str(e)})

@app.post("/translate_sentences")
async def translate_multiple_sentences(request: Request, sentences: List[str] = Form(...), from_lang: str = Form("en"), to_lang: str = Form("zh")):
    """多句翻译"""
    try:
        translated_sentences = translate_sentences(sentences, from_lang, to_lang)
        return templates.TemplateResponse("word_learning.html", {"request": request, "translated_sentences": translated_sentences})
    except Exception as e:
        return templates.TemplateResponse("word_learning.html", {"request": request, "error": str(e)})

from wordreco import eval_new_data

@app.get("/word_recognition", response_class=HTMLResponse)
async def word_recognition_page(request: Request):
    """渲染文字识别页面"""
    return templates.TemplateResponse("word_recognition.html", {"request": request})

@app.post("/recognize_text")
async def recognize_text(image: UploadFile = File(...)):
    """处理上传的图片并返回识别结果"""
    if not image:
        raise HTTPException(status_code=400, detail="No image file uploaded")
    
    # 检查文件类型
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File type not allowed")
        
    try:
        # 调用wordreco.py中的评估函数
        result = await eval_new_data(image)
        
        if result["success"]:
            return {"text": result["text"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/word_analysis", response_class=HTMLResponse)
async def word_analysis_page(request: Request):
    """渲染文本分析页面"""
    if "username" not in request.session:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("word_analysis.html", {"request": request})

@app.post("/analyze_word")
async def analyze_word(request: Request, word: str = Form(...)):
    """单词分析"""
    try:
        # 使用TextBlob进行情感分析
        blob = TextBlob(word)
        sentiment = blob.sentiment
        
        # 计算情感得分和标签
        sentiment_score = round((sentiment.polarity + 1) * 50, 2)  # 转换为0-100的范围
        
        if sentiment.polarity > 0.3:
            sentiment_label = "积极"
        elif sentiment.polarity < -0.3:
            sentiment_label = "消极"
        else:
            sentiment_label = "中性"
            
        # 计算正向和负向得分
        positive_score = round(max(sentiment.polarity, 0) * 100, 2)
        negative_score = round(abs(min(sentiment.polarity, 0)) * 100, 2)
            
        # 计算置信度
        confidence = round(abs(sentiment.polarity) * 100, 2)
        
        return templates.TemplateResponse(
            "word_analysis.html",
            {
                "request": request,
                "word": word,
                "sentiment_analysis": {
                    "score": sentiment_score,
                    "label": sentiment_label,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "confidence": confidence
                }
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "word_analysis.html",
            {"request": request, "error": str(e)}
        )

@app.post("/analyze_text")
async def analyze_text(request: Request, text: str = Form(...)):
    """文本分析"""
    try:
        # 使用TextBlob进行情感分析
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # 计算情感得分和标签
        sentiment_score = round((sentiment.polarity + 1) * 50, 2)
        
        if sentiment.polarity > 0.3:
            sentiment_label = "积极"
        elif sentiment.polarity < -0.3:
            sentiment_label = "消极"
        else:
            sentiment_label = "中性"
            
        # 计算正向和负向得分
        positive_score = round(max(sentiment.polarity, 0) * 100, 2)
        negative_score = round(abs(min(sentiment.polarity, 0)) * 100, 2)
            
        # 计算置信度
        confidence = round(abs(sentiment.polarity) * 100, 2)
        
        # 主题分析
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # 过滤停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        topics = [(word, freq) for word, freq in word_freq.most_common(5) if word not in stop_words]
        
        # 计算主题分析结果
        total_words = len(words)
        topic_analysis = []
        for word, freq in topics:
            weight = round((freq / total_words) * 100, 2)
            relevance = round(weight * (1 + abs(TextBlob(word).sentiment.polarity)), 2)
            topic_analysis.append({
                "keyword": word,
                "weight": weight,
                "frequency": freq,
                "relevance": relevance
            })
        
        return templates.TemplateResponse(
            "word_analysis.html",
            {
                "request": request,
                "text": text,
                "sentiment_analysis": {
                    "score": sentiment_score,
                    "label": sentiment_label,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "confidence": confidence
                },
                "topic_analysis": topic_analysis
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "word_analysis.html",
            {"request": request, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)