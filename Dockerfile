FROM python:3.11

RUN pip install poetry

COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-dev

COPY model.pth /app/model.pth
