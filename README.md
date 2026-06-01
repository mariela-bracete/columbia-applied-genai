# FastAPI Docker NLP App

This project is a FastAPI application containerized with Docker.

## Features

* Bigram text generation endpoint
* spaCy word embedding endpoint
* Dockerized deployment

### Example Embedding Request

POST `/embedding`

```json
{
  "word": "queen"
}```

### Example Response

```json
{
  "word": "queen",
  "embedding": [...]
}
```

## Run Locally

### Build Docker image

```bash
docker build -t gentext-app .
```

### Run Container

```bash
docker run -p 8000:80 gentext-app
```

### Open API docs

http://127.0.0.1:8000/docs




