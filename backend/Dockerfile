# CPU-only torch is a fraction of the size of the default wheel, which pulls in
# the whole CUDA stack. On a 9-input network the GPU build buys nothing and
# costs minutes of cold start.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY rl_agents ./rl_agents

EXPOSE 8000

# One worker: the model is loaded per process and a tic-tac-toe move is ~0.02 ms,
# so extra workers only multiply the memory footprint.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
