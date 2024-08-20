from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from predict import analyze_single_audio, analyze_folder, analyze_long_audio, models, emotion_labels
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


class FolderAnalysis(BaseModel):
    folder_path: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/analyze-single")
async def analyze_single(audio_file: UploadFile = File(...), model_name: str = Form(...)):
    predictions_with_labels = analyze_single_audio(audio_file.file, model_name)
    return JSONResponse(content=predictions_with_labels)


@app.post("/analyze-folder")
async def analyze_folder_api(folder_analysis: FolderAnalysis):
    avg_probs = analyze_folder(folder_analysis.folder_path)
    return JSONResponse(content=avg_probs)


@app.post("/analyze-long")
async def analyze_long(audio_file: UploadFile = File(...)):
    plt_fig = analyze_long_audio(audio_file.file)
    plt_fig.savefig("static/plot.png")
    return {"plot_url": "/static/plot.png"}
