# CNN Image Classification API with FastAPI

A Dockerized REST API built with FastAPI that provides:

- CNN-based image classification trained on CIFAR-10
- Bigram text generation
- Word embeddings using spaCy
- Interactive API documentation via Swagger/OpenAPI

Built as part of Columbia University's Applied Generative AI coursework.

## Features

### REST API Endpoints

- `/generate` – Bigram text generation
- `/embedding` – Text embeddings using spaCy
- `/classify-image` – CNN image classification

### Image Classification

- Convolutional Neural Network (CNN) trained on CIFAR-10
- PyTorch model persistence (`.pth`)
- Image upload and prediction via REST API
- Confidence score returned with prediction

**Supported classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Deployment

- Dockerized application
- FastAPI + Uvicorn server
- Swagger/OpenAPI documentation## Features

## API Documentation

Interactive Swagger UI available through FastAPI.

## Containerization

Application is fully containerized using Docker for consistent deployment.

## Project Structure

```text
app/
├── main.py
├── bigram_model.py
├── cnn_model.py
├── image_classifier.py
├── models/
│   └── cnn_cifar10.pth

helper_lib/
├── checkpoints.py
├── data_loader.py
├── evaluator.py
├── model.py
├── trainer.py
└── utils.py

train_cnn.py
Dockerfile
```

### Example Embedding Request

POST `/embedding`

```json
{
  "word": "queen"
}
```

### Example Response

```json
{
  "word": "queen",
  "embedding": [...]
}
```

### Example Embedding Request

POST `/embedding`

```json
{
  "word": "queen"
}
```

### Example Response

```json
{
  "word": "queen",
  "embedding": [...]
}
```

## Run Locally

### Build Docker Image

```bash
docker build -t fastapi-ml-api .
```

### Run Container

```bash
docker run -p 8000:80 fastapi-ml-api
```

### Open API Documentation

http://localhost:8000/docs

## Technologies Used

- Python
- FastAPI
- PyTorch
- torchvision
- spaCy
- Docker
- Uvicorn
