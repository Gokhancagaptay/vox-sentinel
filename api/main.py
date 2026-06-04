"""
VoxSentinel REST API

Endpoint'ler:
    GET  /                     — Web arayüzü (templates/index.html)
    GET  /health               — Sistem sağlığı (ffmpeg, vosk model, whisper_mode)
    POST /censor               — Sync işleme; sansürlü dosyayı döndür
    POST /censor/async         — Arka plan işleme; job_id döndür
    GET  /jobs/{job_id}        — İş durumu ve indirme URL'si
    GET  /jobs/{job_id}/download — Sansürlü dosyayı indir
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config.banned_words import YASAKLI_KELIMELER
from config.settings import VOSK_MODEL_PATH, WHISPER_MODE
from core.pipeline import run_censorship_pipeline_async

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

app = FastAPI(
    title="VoxSentinel API",
    description="Çok katmanlı Türkçe ses sansür sistemi",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Bellek içi iş deposu — üretim ortamında Redis ile değiştirin
_jobs: dict[str, dict[str, Any]] = {}


# ─── Modeller ────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "processing" | "done" | "error"
    output_url: str | None = None
    error: str | None = None
    segments_censored: int = 0


class HealthResponse(BaseModel):
    status: str
    ffmpeg: bool
    vosk_model: bool
    whisper_mode: str
    banned_word_count: int


# ─── Endpoint'ler ─────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    import shutil
    return HealthResponse(
        status="ok",
        ffmpeg=shutil.which("ffmpeg") is not None,
        vosk_model=Path(VOSK_MODEL_PATH).exists(),
        whisper_mode=WHISPER_MODE,
        banned_word_count=len(YASAKLI_KELIMELER),
    )


@app.post("/censor")
async def censor_sync(file: UploadFile = File(...)) -> FileResponse:
    """Ses dosyasını senkron olarak işler; sansürlü dosyayı döndürür."""
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="vs_in_") as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    output_path = input_path.replace("vs_in_", "vs_out_")

    try:
        result = await run_censorship_pipeline_async(input_path, output_path)
    except Exception as exc:
        _safe_unlink(input_path)
        raise HTTPException(status_code=500, detail=str(exc))

    _safe_unlink(input_path)

    serve_path = output_path if result.censored else input_path
    return FileResponse(
        serve_path,
        media_type="audio/wav",
        filename=f"censored_{file.filename}",
    )


@app.post("/censor/async", response_model=JobStatus)
async def censor_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JobStatus:
    """Ses dosyasını arka planda işler; job_id ile durum sorgulanabilir."""
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix=f"vs_{job_id}_in_") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(f"vs_{job_id}_in_", f"vs_{job_id}_out_")

    _jobs[job_id] = {"status": "pending", "output_path": output_path}
    background_tasks.add_task(_process_job, job_id, input_path, output_path)

    return JobStatus(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job bulunamadı.")
    job = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        output_url=f"/jobs/{job_id}/download" if job["status"] == "done" else None,
        error=job.get("error"),
        segments_censored=job.get("segments_censored", 0),
    )


@app.get("/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    if job_id not in _jobs or _jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=404, detail="Dosya henüz hazır değil.")
    return FileResponse(_jobs[job_id]["output_path"], media_type="audio/wav")


# ─── Yardımcılar ──────────────────────────────────────────────────

async def _process_job(job_id: str, input_path: str, output_path: str) -> None:
    _jobs[job_id]["status"] = "processing"
    try:
        result = await run_censorship_pipeline_async(input_path, output_path)
        _jobs[job_id].update({
            "status": "done",
            "segments_censored": len(result.final_censor_segments),
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
        _safe_unlink(input_path)


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
