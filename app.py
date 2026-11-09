"""Osprey: a dry bulk market dashboard with an AI assistant.

The opening page (/) shows spot and FFA prices plus the latest market news,
with an "Ask Osprey AI" button that opens the chat (/chat). The chat sends
each question to /api/answer and shows the RAG-grounded answer with its
sources; a question can be stopped through /api/cancel. Each news item opens
in Osprey (/article/<id>) with a side chat about that article
(/api/article/<id>/answer) and links to articles from the same and previous days.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""
import io
import threading
from pathlib import Path

from flask import Flask, Response, abort, redirect, render_template, request, send_from_directory, url_for
from PIL import Image

import prices
import rag_engine

app = Flask(__name__)

SITE_NAME = "Osprey"
BASE_DIR = Path(__file__).resolve().parent

# Frames of the "answer is being generated" animation: only the numbered files
# (1.png, 2.png, ..., 10.png), in numeric order. Other images in the folder, like
# Ship-logo.png, are not frames.
LOADING_DIR = BASE_DIR / "loading"

# The site logo.
LOGO_PATH = LOADING_DIR / "Ship-logo.png"
_logo_cache: dict[float, bytes] = {}

SUGGESTED_QUESTIONS = [
    "How is the current dry bulk market?",
    "Any recent capesize newbuilding orders?",
    "What drove Panamax rates in March 2025?",
    "What is happening with Simandou iron ore?",
]
ARTICLE_SUGGESTIONS = [
    "Summarise this article in three points",
    "Which companies and ships are mentioned?",
    "What does this mean for freight rates?",
]
LOADING_FRAMES = sorted(
    (p.name for p in LOADING_DIR.glob("*.png") if p.stem.isdigit()),
    key=lambda name: int(Path(name).stem),
)

# Kick off the (slow, one-time) data load / index build in the background as
# soon as the app starts, instead of blocking the first request.
rag_engine.engine.build_async()


@app.context_processor
def inject_globals():
    return {
        "site_name": SITE_NAME,
        "ollama_model": rag_engine.OLLAMA_MODEL,
        "loading_frames": [url_for("loading_frame", name=n) for n in LOADING_FRAMES],
    }


@app.route("/loading/<name>")
def loading_frame(name):
    return send_from_directory(LOADING_DIR, name, max_age=86400)


def crop_logo(path: Path) -> bytes:
    """Trim the logo's transparent margin and centre it on a square, so it fills its badge."""
    im = Image.open(path).convert("RGBA")
    box = im.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox() or (0, 0, *im.size)
    im = im.crop(box)
    side = int(max(im.size) * 1.08)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


@app.route("/logo.png")
def logo():
    if not LOGO_PATH.exists():
        abort(404)
    key = LOGO_PATH.stat().st_mtime   # re-crop if the file is replaced
    if key not in _logo_cache:
        _logo_cache.clear()
        _logo_cache[key] = crop_logo(LOGO_PATH)
    return Response(_logo_cache[key], mimetype="image/png", headers={"Cache-Control": "max-age=300"})


@app.route("/", methods=["GET"])
def home():
    engine = rag_engine.engine
    return render_template(
        "home.html",
        prices=prices.get_prices(),
        engine_ready=engine.ready, engine_status=engine.status,
        market_news=engine.latest_articles(12, market_only=True) if engine.ready else [],
        all_news=engine.latest_articles(12) if engine.ready else [],
    )


def parse_n_sources() -> int:
    try:
        return max(5, min(10, int(request.args.get("n", "8"))))
    except ValueError:
        return 8


@app.route("/chat", methods=["GET"])
def chat():
    """The Osprey AI chat page. Its script sends each question to /api/answer.
    A ?q= parameter is asked straight away."""
    return render_template("chat.html", n_sources=parse_n_sources(), suggestions=SUGGESTED_QUESTIONS)


@app.route("/article/<int:article_id>", methods=["GET"])
def article(article_id):
    """One article in Osprey: its text, a link to the original, a chat about it,
    and the articles from the same and previous days."""
    engine = rag_engine.engine
    if not engine.ready:
        return render_template("article.html", article=None, related=[],
                               engine_status=engine.status, suggestions=[])
    found = engine.get_article(article_id)
    if found is None:
        abort(404)
    return render_template("article.html", article=found, related=engine.related_articles(article_id),
                           engine_status=engine.status, suggestions=ARTICLE_SUGGESTIONS)


@app.route("/ask", methods=["GET"])
def ask():
    """Old answer-page address; questions now live in the chat."""
    return redirect(url_for("chat", **request.args))


# Questions being answered right now, by the page's request id, so /api/cancel
# can stop them. A cancel that arrives first leaves its set Event behind, and
# the answer request then stops before it starts.
_running: dict[str, threading.Event] = {}
_running_lock = threading.Lock()


def _cancel_event(rid: str) -> threading.Event:
    with _running_lock:
        return _running.setdefault(rid, threading.Event())


def _run_question(work, template: str):
    """Run `work(cancel)` for the question in the request (q, rid) and return the
    rendered `template` as {html}, or {cancelled} / {error}."""
    query = request.args.get("q", "").strip()
    rid = request.args.get("rid", "")
    if not query or not rid:
        return {"error": "Missing question."}, 400
    if not rag_engine.engine.ready:
        return {"error": "The search engine is still warming up."}, 503

    cancel = _cancel_event(rid)
    try:
        result = work(query, cancel)
    except rag_engine.AnswerCancelled:
        return {"cancelled": True}
    except Exception as exc:  # shown on the page rather than as a 500 page
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        with _running_lock:
            _running.pop(rid, None)
    return {"html": render_template(template, result=result)}


@app.route("/api/answer", methods=["GET"])
def api_answer():
    n_sources = parse_n_sources()
    return _run_question(
        lambda q, cancel: rag_engine.engine.answer(q, n_sources=n_sources, cancel=cancel), "_result.html")


@app.route("/api/article/<int:article_id>/answer", methods=["GET"])
def api_article_answer(article_id):
    return _run_question(
        lambda q, cancel: rag_engine.engine.answer_article(article_id, q, cancel=cancel), "_article_answer.html")


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    rid = request.form.get("rid", "")
    if rid:
        _cancel_event(rid).set()
    return {"ok": bool(rid)}


@app.route("/status")
def status():
    return {"ready": rag_engine.engine.ready, "status": rag_engine.engine.status}


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
