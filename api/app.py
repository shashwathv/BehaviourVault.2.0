import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import json
import re
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from fastapi import FastAPI, Header, HTTPException, Request, Depends, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tflite_runtime.interpreter import Interpreter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(
    force_terminal=True,
    width=int(os.environ.get('CONSOLE_WIDTH', '100')),
    log_path=False,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_PATH = os.path.join(BASE_DIR, "models", "behavior_model.tflite")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
INPUT_DETAILS  = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()


MEAN = [604.4399, 0.5071, 451.8634, 179.5553, 0.0493]
STD  = [191.6922, 0.0997, 144.1839,  78.1840, 0.0192]


EWMA_ALPHA               = 0.15
WARMUP_SESSIONS          = 3
ANOMALY_THRESHOLD        = 0.8    
SEVERE_ANOMALY_THRESHOLD = 0.95  
ELEVATED_THRESHOLD       = 0.5
GLOBAL_WEIGHT            = 0.3   
BASELINE_WEIGHT          = 0.7  
BASELINE_NORMALIZER      = 3.0
SUMMARY_EVERY            = 25

BASELINE_DIR = os.path.join(BASE_DIR, "baselines")
os.makedirs(BASELINE_DIR, exist_ok=True)

FEATURE_KEYS = [
    'keystroke_avg_ms',
    'touch_pressure_avg',
    'swipe_avg_px_per_sec',
    'scroll_rhythm_ms',
    'accelerometer_avg_variance',
]

LEVEL_STYLE = {
    "NORMAL":   ("green",  "✓"),
    "ELEVATED": ("yellow", "!"),
    "ANOMALY":  ("red",    "✗"),
}

API_KEYS = {
    k.strip() for k in os.environ.get('BV_API_KEYS', '').split(',') if k.strip()
}



class PredictInput(BaseModel):
    userId:  Optional[str] = None
    user_id: Optional[str] = None
    keystroke_avg_ms:           float = Field(..., ge=0, le=5000)
    touch_pressure_avg:         float = Field(..., ge=0, le=1)
    swipe_avg_px_per_sec:       float = Field(..., ge=0, le=3000)
    scroll_rhythm_ms:           float = Field(..., ge=0, le=5000)
    accelerometer_avg_variance: float = Field(..., ge=0, le=10)

    def user(self) -> str:
        return self.userId or self.user_id or 'anonymous'

    def features(self) -> list[float]:
        return [
            self.keystroke_avg_ms,
            self.touch_pressure_avg,
            self.swipe_avg_px_per_sec,
            self.scroll_rhythm_ms,
            self.accelerometer_avg_variance,
        ]



STATS_LOCK = threading.Lock()
STATS = {
    'started_at':   time.time(),
    'request_id':   0,
    'total':        0,
    'NORMAL':       0,
    'ELEVATED':     0,
    'ANOMALY':      0,
    'errors':       0,
    'unauthorized': 0,
    'users':        set(),
    'latencies':    deque(maxlen=200),
}
RECENT = deque(maxlen=50)



_USER_ID_RE = re.compile(r'[^A-Za-z0-9_\-]')

def safe_user_id(user_id: str) -> str:
    return _USER_ID_RE.sub('', str(user_id)) or 'anonymous'

def get_baseline_path(user_id: str) -> str:
    return os.path.join(BASELINE_DIR, f"{safe_user_id(user_id)}.json")

def load_baseline(user_id: str):
    path = get_baseline_path(user_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def save_baseline(user_id: str, baseline: dict):
    baseline["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(get_baseline_path(user_id), "w") as f:
        json.dump(baseline, f)

def update_ewma_baseline(user_id, current_features, existing):
    if existing is None:
        new_values = list(current_features)
        session_count = 1
    else:
        old_values = existing["values"]
        new_values = [
            EWMA_ALPHA * curr + (1 - EWMA_ALPHA) * old
            for curr, old in zip(current_features, old_values)
        ]
        session_count = existing.get("session_count", 0) + 1
    baseline = {"values": new_values, "session_count": session_count}
    save_baseline(user_id, baseline)
    return baseline

def score_against_baseline(raw, baseline_values):
    deviations = [abs(raw[i] - baseline_values[i]) / STD[i] for i in range(len(raw))]
    return float(np.mean(deviations))


def impute_for_global_model(raw):
    imputed_features = []
    new_raw = list(raw)
    for i, v in enumerate(raw):
        if v == 0:
            new_raw[i] = MEAN[i]
            imputed_features.append(FEATURE_KEYS[i])
    return new_raw, imputed_features

def run_tflite(raw):
    normalized = [(raw[i] - MEAN[i]) / STD[i] for i in range(5)]
    input_data = np.array([normalized], dtype=np.float32)
    interpreter.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    interpreter.invoke()
    return float(interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])[0][0])

def classify_level(score):
    if score > ANOMALY_THRESHOLD:    return 'ANOMALY'
    if score > ELEVATED_THRESHOLD:   return 'ELEVATED'
    return 'NORMAL'


def get_client_ip(request: Request) -> str:
    return (request.headers.get('CF-Connecting-IP')
            or request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or (request.client.host if request.client else 'unknown'))

def ts():
    return datetime.now().strftime('%H:%M:%S')

def print_banner():
    auth_status = (
        f"[green bold]ENABLED[/] ({len(API_KEYS)} key(s))"
        if API_KEYS else "[red bold]DISABLED[/] — set BV_API_KEYS!"
    )

    info = Table(show_header=False, box=None, padding=(0, 1))
    info.add_column(style="dim", min_width=22)
    info.add_column()
    info.add_row("Status",             "[green bold]RUNNING[/]")
    info.add_row("Framework",          "FastAPI + uvicorn")
    info.add_row("Port",               "5000")
    info.add_row("Model",              os.path.relpath(MODEL_PATH, BASE_DIR))
    info.add_row("API key auth",       auth_status)
    info.add_row("Warmup sessions",    str(WARMUP_SESSIONS))
    info.add_row("Anomaly threshold",  f"score > {ANOMALY_THRESHOLD}")
    info.add_row("Freeze baseline",    f"score > {SEVERE_ANOMALY_THRESHOLD}")
    info.add_row("EWMA α",             str(EWMA_ALPHA))
    info.add_row("Score blend",        f"{GLOBAL_WEIGHT}·global + {BASELINE_WEIGHT}·baseline")
    info.add_row("OpenAPI docs",       "/docs")
    console.print()
    console.print(Panel(info,
                        title="[bold cyan]🛡  BehaviorVault 2.0 API[/]",
                        title_align="left",
                        border_style="cyan",
                        box=box.HEAVY))

def print_request_panel(req_id, ip, user_id, raw, result, latency_ms):
    color, icon = LEVEL_STYLE[result['level']]

    body = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    body.add_column(style="dim", min_width=32)
    body.add_column(justify="right")

    body.add_row("[bold cyan]INPUT[/]", "")
    for k, v in zip(FEATURE_KEYS, raw):
        body.add_row(f"  {k}", f"{v}")

    body.add_row("", "")

    body.add_row("[bold cyan]OUTPUT[/]", "")
    body.add_row("  confidence_pct", f"[bold]{result['confidence_pct']}%[/]")
    body.add_row("  global_score",   f"{result['global_score']:.4f}")
    bs = (f"{result['baseline_score']:.4f}"
          if result['baseline_score'] is not None else "[dim]—[/]")
    body.add_row("  baseline_score", bs)
    body.add_row("  combined_score", f"[bold]{result['score']:.4f}[/]")
    body.add_row("  session_count",  str(result['session_count']))
    if result['warmup_remaining'] > 0:
        body.add_row("  warmup_remaining",
                     f"[yellow]{result['warmup_remaining']} session(s)[/]")
    body.add_row("  using_baseline", "yes" if result['using_baseline'] else "no")
    if result.get('imputed_features'):
        imp = ", ".join(f.replace('_avg', '').replace('_ms', '').replace('_per_sec', '')
                        for f in result['imputed_features'])
        body.add_row("  imputed (missing)", f"[yellow]{imp}[/]")
    body.add_row("  decision",       f"[{color} bold]{icon} {result['level']}[/]")

    title = (f"[bold]#{req_id}[/]  ·  [dim]{ts()}[/]  ·  "
             f"user=[bold]{user_id}[/]  ·  [dim]{ip}[/]  ·  {latency_ms}ms")
    console.print(Panel(body, title=title, title_align="left",
                        border_style=color, box=box.ROUNDED))

def print_summary():
    with STATS_LOCK:
        uptime = int(time.time() - STATS['started_at'])
        h, rem = divmod(uptime, 3600); m, s = divmod(rem, 60)
        total = STATS['total']
        rate  = (STATS['ANOMALY'] / total * 100) if total else 0
        users = len(STATS['users'])
        lats  = sorted(STATS['latencies'])
        p50   = lats[len(lats)//2]        if lats else 0
        p95   = lats[int(len(lats)*0.95)] if lats else 0
        norm, elev, anom = STATS['NORMAL'], STATS['ELEVATED'], STATS['ANOMALY']
        errs, unauth = STATS['errors'], STATS['unauthorized']

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", min_width=18)
    t.add_column(justify="right")
    t.add_row("Uptime",            f"{h:02d}h {m:02d}m {s:02d}s")
    t.add_row("Total requests",    str(total))
    t.add_row("[green]Normal[/]",  f"[green]{norm}[/]")
    t.add_row("[yellow]Elevated[/]", f"[yellow]{elev}[/]")
    t.add_row("[red]Anomaly[/]",   f"[red]{anom}[/]")
    t.add_row("Errors (4xx)",      f"[red]{errs}[/]"  if errs   else "0")
    t.add_row("Unauthorized 401",  f"[red]{unauth}[/]" if unauth else "0")
    t.add_row("Unique users",      str(users))
    t.add_row("Anomaly rate",      f"{rate:.1f}%")
    t.add_row("Latency p50",       f"{p50}ms")
    t.add_row("Latency p95",       f"{p95}ms")

    console.print(Panel(t,
                        title="[bold magenta]📊  SUMMARY[/]",
                        title_align="left",
                        border_style="magenta",
                        box=box.DOUBLE))

def print_compact(ip, method, path, status_code, extra=""):
    color = "green" if 200 <= status_code < 300 else "red" if status_code >= 400 else "yellow"
    console.print(
        f"[dim]{ts()}[/]  [{color}]{status_code}[/]  "
        f"[bold]{method}[/] {path}  [dim]{ip}[/]  [dim]{extra}[/]"
    )

def print_error(ip, status_code, msg):
    console.print(
        f"[dim]{ts()}[/]  [red bold]✗ {status_code}[/]  [dim]{ip}[/]  [red]{msg}[/]"
    )



def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias='X-API-Key'),
):
    """Reject if no key configured-globally → dev mode (allowed).
    Reject 401 if API_KEYS set but key missing/invalid."""
    if not API_KEYS:
        return  
    if not x_api_key or x_api_key not in API_KEYS:
        ip = get_client_ip(request)
        print_error(ip, 401, f"unauthorized attempt at {request.url.path}")
        with STATS_LOCK:
            STATS['unauthorized'] += 1
        raise HTTPException(
            status_code=401,
            detail='Invalid or missing API key',
            headers={"WWW-Authenticate": "X-API-Key"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print_banner()
    if not API_KEYS:
        console.print(
            "[red bold]⚠  WARNING:[/] BV_API_KEYS is empty — API is unprotected. "
            "Set this env var in Coolify to require X-API-Key.\n"
        )
    yield


app = FastAPI(
    title="BehaviorVault 2.0 API",
    version="4.0",
    description=(
        "Mobile behavioral anomaly detection. "
        "All endpoints except /health require X-API-Key header."
    ),
    docs_url="/docs",   
    redoc_url=None,
    lifespan=lifespan,
)



@app.exception_handler(RequestValidationError)
def on_validation_error(request: Request, exc: RequestValidationError):
    ip = get_client_ip(request)
    fields = [".".join(str(p) for p in e["loc"]) + f": {e['msg']}" for e in exc.errors()]
    with STATS_LOCK:
        STATS['errors'] += 1
    print_error(ip, 422, f"validation failed: {fields}")
    return JSONResponse(
        status_code=422,
        content={'error': 'validation failed', 'details': exc.errors()},
    )



@app.get('/health')
def health():
    return {
        'status':          'ok',
        'model_loaded':    True,
        'warmup_sessions': WARMUP_SESSIONS,
        'feature_keys':    FEATURE_KEYS,
        'auth_enabled':    bool(API_KEYS),
    }


@app.post('/predict', dependencies=[Depends(require_api_key)])
def predict(payload: PredictInput, request: Request):
    start = time.perf_counter()
    ip = get_client_ip(request)

    with STATS_LOCK:
        STATS['request_id'] += 1
        req_id = STATS['request_id']

    user_id = payload.user()
    raw = payload.features()

    #Global model (with zero-imputation)
    raw_for_model, imputed_features = impute_for_global_model(raw)
    global_score = run_tflite(raw_for_model)

    #Personal baseline 
    baseline = load_baseline(user_id)
    prior_session_count = baseline['session_count'] if baseline else 0
    baseline_score = (
        score_against_baseline(raw, baseline['values']) if baseline else None
    )

    using_baseline = (
        baseline_score is not None and prior_session_count >= WARMUP_SESSIONS
    )
    if using_baseline:
        norm_baseline = min(baseline_score / BASELINE_NORMALIZER, 1.0)
        combined_score = (GLOBAL_WEIGHT * global_score
                          + BASELINE_WEIGHT * norm_baseline)
    else:
        combined_score = global_score

    is_anomaly = combined_score > ANOMALY_THRESHOLD
    freeze_baseline = combined_score > SEVERE_ANOMALY_THRESHOLD

    if not freeze_baseline:
        updated = update_ewma_baseline(user_id, raw, baseline)
        session_count = updated['session_count']
    else:
        session_count = prior_session_count

    level = classify_level(combined_score)
    latency_ms = int((time.perf_counter() - start) * 1000)
    confidence_pct = int(round((1.0 - combined_score) * 100))

    result = {
        'user_id':           user_id,
        'confidence_pct':    confidence_pct,
        'score':             round(combined_score, 4),
        'global_score':      round(global_score, 4),
        'baseline_score':    round(baseline_score, 4) if baseline_score is not None else None,
        'using_baseline':    using_baseline,
        'warmup_remaining':  max(0, WARMUP_SESSIONS - session_count),
        'session_count':     session_count,
        'is_anomaly':        is_anomaly,
        'level':             level,
        'baseline_updated':  not freeze_baseline,
        'imputed_features':  imputed_features,   
    }

    with STATS_LOCK:
        STATS['total'] += 1
        STATS[level] += 1
        STATS['users'].add(user_id)
        STATS['latencies'].append(latency_ms)
        total_now = STATS['total']
        RECENT.append({
            'id': req_id, 'ts': time.time(), 'ip': ip, 'user': user_id,
            'level': level, 'score': result['score'], 'latency_ms': latency_ms,
        })

    print_request_panel(req_id, ip, user_id, raw, result, latency_ms)
    if total_now % SUMMARY_EVERY == 0:
        print_summary()

    return result


@app.get('/baseline/{user_id}', dependencies=[Depends(require_api_key)])
def get_baseline_endpoint(user_id: str, request: Request):
    ip = get_client_ip(request)
    baseline = load_baseline(user_id)
    if baseline is None:
        print_compact(ip, "GET", f"/baseline/{user_id}", 404, "no baseline")
        raise HTTPException(status_code=404, detail='No baseline found')
    print_compact(ip, "GET", f"/baseline/{user_id}", 200,
                  f"sessions={baseline['session_count']}")
    return {
        'user_id':       user_id,
        'session_count': baseline['session_count'],
        'baseline':      dict(zip(FEATURE_KEYS, baseline['values'])),
        'updated_at':    baseline.get('updated_at'),
    }


@app.delete('/baseline/{user_id}', dependencies=[Depends(require_api_key)])
def reset_baseline_endpoint(user_id: str, request: Request):
    ip = get_client_ip(request)
    path = get_baseline_path(user_id)
    if os.path.exists(path):
        os.remove(path)
        print_compact(ip, "DELETE", f"/baseline/{user_id}", 200, "reset")
        return {'message': f'Baseline for {user_id} reset.'}
    print_compact(ip, "DELETE", f"/baseline/{user_id}", 404, "no baseline")
    raise HTTPException(status_code=404, detail='No baseline found')


@app.get('/stats', dependencies=[Depends(require_api_key)])
def get_stats(request: Request):
    print_compact(get_client_ip(request), "GET", "/stats", 200, "summary requested")
    print_summary()
    with STATS_LOCK:
        uptime = time.time() - STATS['started_at']
        return {
            'uptime_seconds':  round(uptime, 2),
            'total_requests':  STATS['total'],
            'normal':          STATS['NORMAL'],
            'elevated':        STATS['ELEVATED'],
            'anomaly':         STATS['ANOMALY'],
            'errors':          STATS['errors'],
            'unauthorized':    STATS['unauthorized'],
            'unique_users':    len(STATS['users']),
            'anomaly_rate':    (STATS['ANOMALY'] / max(1, STATS['total'])),
            'recent_requests': list(RECENT),
        }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level='warning')