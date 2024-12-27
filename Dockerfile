FROM python:3.12.8-slim

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# RUN pip install --no-cache-dir nucs
# RUN pip install rich

COPY nucs /nucs

CMD ["python",  "-m", "nucs.examples.quasigroup"]