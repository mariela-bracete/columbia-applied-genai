from io import BytesIO
from typing import Union

import spacy
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.energy_model import generate_energy_image
from app.image_classifier import classify_image
from app.mnist_gan import generate_mnist_digit
from app.diffusion import generate_diffusion_image

app = FastAPI()
nlp = spacy.load("en_core_web_md")
class EmbeddingRequest(BaseModel):
    word: str

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {"generated_text": generated_text}

@app.post(
    "/generate-energy-image",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    },
)
async def generate_energy_image_endpoint():
    return generate_energy_image()

@app.post(
    "/generate-diffusion-image",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    },
)
async def generate_diffusion_image_endpoint():
    return Response(
        content=generate_diffusion_image(),
        media_type="image/png",
    )

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    token = nlp(request.word)

    if not token.has_vector:
        return {
            "word": request.word,
            "error": "No embedding found for this word."
        }

    return {
        "word": request.word,
        "embedding": token.vector.tolist(),
        "embedding_length": len(token.vector)
    }

@app.post("/classify-image")
async def classify_uploaded_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    prediction = classify_image(image)

    return {
        "filename": file.filename,
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
    }

@app.post(
    "/generate-mnist-digit",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    }
)
async def generate_mnist_digit_endpoint():
    return generate_mnist_digit()