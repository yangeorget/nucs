FROM python:3.12.8-slim

ARG NUCS_VERSION
#ENV PYTHON_CMD="python -m $NUCS_MODULE"

# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

RUN pip install --no-cache-dir nucs==$NUCS_VERSION
RUN pip install rich

COPY nucs /nucs

CMD python -m nucs.examples.queens