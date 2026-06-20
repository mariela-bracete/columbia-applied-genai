# CNN Image Classification API with FastAPI

A Dockerized FastAPI application exposing multiple machine learning capabilities through REST APIs.

## Features

### Text Generation
- Bigram language model endpoint
- Generates text continuations from a seed prompt

### Word Embeddings
- spaCy-based embedding endpoint
- Returns vector representations of text

### Image Classification
- CNN image classifier trained on the CIFAR-10 dataset
- Supports image upload through FastAPI
- Returns predicted class and confidence score

**Supported classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

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