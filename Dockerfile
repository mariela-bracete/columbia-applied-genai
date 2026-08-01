# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Install curl and certificates for the uv installer
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure uv is available
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy dependency files first to take advantage of Docker layer caching
COPY pyproject.toml uv.lock /code/

# Install locked dependencies
RUN uv sync --frozen

# Copy the application, saved models, and supporting code
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./checkpoints /code/checkpoints

# Copy training scripts
COPY ./train_cnn.py /code/train_cnn.py
COPY ./train_energy.py /code/train_energy.py
COPY ./train_diffusion.py /code/train_diffusion.py
COPY ./train_llm.py /code/train_llm.py
COPY ./train_rl.py /code/train_rl.py

# Run the FastAPI application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]