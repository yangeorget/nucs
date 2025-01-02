FROM python:3.12.8-slim

ARG NUCS_MODULE
ENV PYTHON_CMD="python -m $NUCS_MODULE"

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# RUN pip install --no-cache-dir nucs
# RUN pip install rich

COPY nucs /nucs

CMD $PYTHON_CMD