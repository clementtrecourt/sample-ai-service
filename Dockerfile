FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
# Création d'un utilisateur non-root pour la sécurité
RUN useradd -u 10001 appuser
USER 10001

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser main.py .

ENV PATH=/home/appuser/.local/bin:$PATH
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
