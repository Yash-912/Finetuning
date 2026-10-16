# GPU serving image for the injection-detector proxy.
# Build:  docker build -t injection-detector .
# Run:    docker run --gpus all -p 8080:8080 -e LLM_API_KEY=... injection-detector
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY middleware/requirements.txt middleware/requirements.txt
RUN pip install --no-cache-dir -r middleware/requirements.txt

COPY middleware/ middleware/
COPY scripts/review_queue.py scripts/review_queue.py
COPY models/qwen-injection-detector/best/ models/qwen-injection-detector/best/

ENV HOST=0.0.0.0 PORT=8080 REVIEW_QUEUE_PATH=/data/review_queue.db
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

CMD ["python", "-m", "middleware.app"]
