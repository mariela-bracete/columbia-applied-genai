# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
# Set the working directory
WORKDIR /code
# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/
# Install dependencies using uv
RUN uv sync --frozen
# Copy the application code
COPY ./app /code/app
# Copy helper library and training script
COPY ./helper_lib /code/helper_lib
COPY ./train_cnn.py /code/train_cnn.py
# Command to run the application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
# Copy the checkpoints directory for Energy and Diffusion APIs
COPY ./checkpoints /code/checkpoints
# Copy the training scripts for Energy and Diffusion models
COPY ./train_energy.py /code/train_energy.py
COPY ./train_diffusion.py /code/train_diffusion.py