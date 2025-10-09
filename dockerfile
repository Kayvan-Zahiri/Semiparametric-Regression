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
EXPOSE 8050
CMD ["python", "main.py"]
