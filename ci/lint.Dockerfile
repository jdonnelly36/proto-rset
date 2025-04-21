FROM python:3.8.18-alpine

RUN pip install --upgrade pip
COPY env/requirements-lint.txt requirements-lint.txt
RUN pip install -r requirements-lint.txt