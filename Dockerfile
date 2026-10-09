FROM python:3.13-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
COPY . /webapp
WORKDIR /webapp
RUN uv venv
RUN uv pip install -r requirements.txt

FROM python:3.13-slim-bookworm
COPY --from=builder /webapp /webapp
ENV PATH="/webapp/.venv/bin:$PATH"
WORKDIR /webapp
# Cloud Run sends traffic to $PORT (8080 unless the service sets another
# container port). Fall back to 8050 for local `docker run` without PORT.
EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8050} --timeout 120 main:server"]
