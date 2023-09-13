# Use an official Python 3.11 image as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Copy the model into the container
COPY model_cnn/saved_models/densenet121/best_model.pth /app/saved_models/densenet121/best_model.pth

# Install poetry
RUN pip install poetry

# Disable virtual env creation by poetry as containers are already isolated
RUN poetry config virtualenvs.create false

# Install dependencies using poetry
RUN poetry install --no-dev

# CMD is specified when you run the Docker container
