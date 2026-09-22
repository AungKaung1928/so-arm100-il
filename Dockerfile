# CPU-only. Runs the tests that need no GL context; the end-to-end smoke and
# the image test skip inside the container.
#   docker build -t so-arm100-il .
#   docker run --rm so-arm100-il
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY so_arm100_il ./so_arm100_il
COPY tests ./tests
COPY scripts ./scripts
COPY train_bc.py dagger.py eval_policy.py scaling.py verify.sh ./

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "tests"]
