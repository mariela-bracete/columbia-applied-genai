# Multi-Model Generative AI API

A Dockerized REST API built with **FastAPI** that exposes multiple machine learning and generative AI models through a unified HTTP interface.

Developed as part of Columbia University's **Applied Generative AI** coursework, this project combines classical natural language processing, transformer-based language models, computer vision, generative adversarial networks (GANs), Energy-Based Models (EBMs), Diffusion Models, and reinforcement learning into a single deployable FastAPI application.

---

# Features

The API currently supports seven AI capabilities:

- Bigram text generation
- Word embeddings using spaCy
- CNN image classification (CIFAR-10)
- MNIST handwritten digit generation using a Wasserstein GAN (WGAN)
- CIFAR-10 image generation using an Energy-Based Model (EBM)
- CIFAR-10 image generation using a Diffusion Model
- GPT-2 question answering using supervised fine-tuning followed by reinforcement learning post-training

Interactive API documentation is automatically available through Swagger/OpenAPI.

---

# API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /generate` | Generate text using the trained Bigram language model |
| `POST /embedding` | Return a spaCy embedding vector for a word |
| `POST /classify-image` | Classify an uploaded image using the trained CNN |
| `POST /generate-mnist-digit` | Generate a handwritten MNIST digit using the trained WGAN |
| `POST /generate-energy-image` | Generate a CIFAR-10 style image using an Energy-Based Model |
| `POST /generate-diffusion-image` | Generate a CIFAR-10 style image using a Diffusion Model |
| `POST /generate_with_llm` | Generate a question-answer response using the RL-post-trained GPT-2 language model |

---

# Results

The following screenshots illustrate the functionality of each model exposed through the FastAPI application.

## FastAPI Swagger UI

The API provides an interactive Swagger/OpenAPI interface that allows each endpoint to be tested directly from the browser.

![FastAPI Swagger UI](assets/00_FastAPI_Swagger_UI.png)

---

## Word Embeddings

The `/embedding` endpoint returns dense semantic vector representations using pretrained spaCy embeddings.

![Word Embeddings](assets/01_word_embeddings.png)

---

## CNN Image Classification

The `/classify-image` endpoint predicts the CIFAR-10 class of an uploaded image and returns both the predicted label and confidence score.

![CNN Classification](assets/02_image_classification_airplane.png)

---

## Wasserstein GAN (MNIST)

The `/generate-mnist-digit` endpoint generates new handwritten digits learned from the MNIST dataset.

![WGAN MNIST Generation](assets/03_wgan_mnist_generation.png)

---

## Energy-Based Model

The `/generate-energy-image` endpoint synthesizes CIFAR-10 style images using Langevin Dynamics sampling.

![Energy-Based Model](assets/04_energy_based_generation.png)

---

## Diffusion Model

The `/generate-diffusion-image` endpoint produces CIFAR-10 style images using an iterative denoising diffusion process.

![Diffusion Model](assets/05_diffusion_image_generation.png)

---

## GPT-2 Language Model (Supervised Fine-Tuning + Reinforcement Learning)

The `/generate_with_llm` endpoint generates responses using a GPT-2 model that was first supervised fine-tuned and then post-trained using reinforcement learning with a custom reward function encouraging a consistent response format.

![GPT-2 RL Response](assets/06_gpt_rl_text_generation.png)

---

## Reinforcement Learning Training

The figure below shows the average reward obtained during reinforcement learning post-training. The increasing reward demonstrates that the model successfully learned to maximize the specified reward function while preserving coherent text generation.

![RL Training Rewards](rl_training_rewards.png)

---

# Bigram Language Model

The project includes a simple statistical language model capable of generating text based on learned bigram probabilities.

Features include:

- Token probability estimation
- Text generation
- REST API endpoint

---

# Word Embeddings

The API exposes pretrained **spaCy** word embeddings.

Features include:

- Dense vector representations
- Semantic similarity support
- REST API endpoint

---

# CNN Image Classification

The image classification endpoint uses a Convolutional Neural Network trained on the CIFAR-10 dataset.

Features include:

- PyTorch model persistence
- Image upload through REST API
- Predicted class
- Confidence score

Supported classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

# MNIST Digit Generation (WGAN)

The project includes a Wasserstein Generative Adversarial Network (WGAN) capable of generating synthetic handwritten digits.

Features include:

- Generator network
- Critic network
- Wasserstein loss
- PyTorch checkpoints
- FastAPI endpoint returning PNG images

Calling

```
POST /generate-mnist-digit
```

returns a newly generated handwritten digit.

---

# Energy-Based Model (EBM)

The project includes an Energy-Based Model trained on CIFAR-10 for image generation.

Features include:

- Energy-based learning objective
- Langevin Dynamics sampling
- PyTorch checkpoints
- FastAPI endpoint returning PNG images

Calling

```
POST /generate-energy-image
```

returns a synthesized CIFAR-10 style image generated through energy minimization.

---

# Diffusion Model

The project includes a Denoising Diffusion Model trained on CIFAR-10.

Features include:

- Forward diffusion process
- Reverse denoising sampling
- PyTorch checkpoints
- FastAPI endpoint returning PNG images

Calling

```
POST /generate-diffusion-image
```

returns a synthesized CIFAR-10 style image generated through iterative denoising.

---

# GPT-2 Language Model with Reinforcement Learning

The project includes a GPT-2 language model that is first supervised fine-tuned on a question-answering dataset and then post-trained using reinforcement learning to encourage responses that follow a desired conversational format through reward-based optimization.

Features include:

- Supervised fine-tuning using Hugging Face Transformers
- LoRA parameter-efficient fine-tuning
- Reinforcement learning (RL) post-training
- Custom reward function
- Model checkpointing
- FastAPI REST endpoint

Calling

```
POST /generate_with_llm
```

returns a generated response from the RL-post-trained GPT-2 model.

---

# Project Structure

```text
app/
├── __init__.py
├── main.py
├── bigram_model.py
├── cnn_model.py
├── diffusion.py
├── energy_model.py
├── image_classifier.py
├── llm_model.py
├── mnist_gan.py
├── models/
│   ├── cnn_cifar10.pth
│   ├── mnist_wgan_generator.pt
│   ├── gpt2_qa/
│   └── gpt2_rl/

helper_lib/
├── checkpoints.py
├── data_loader.py
├── evaluator.py
├── generator.py
├── model.py
├── trainer.py
└── utils.py

checkpoints/
├── cnn/
├── diffusion/
├── energy/
└── mnist_gan/

assets/
├── 00_FastAPI_Swagger_UI.png
├── 01_word_embeddings.png
├── 02_image_classification_airplane.png
├── 03_wgan_mnist_generation.png
├── 04_energy_based_generation.png
├── 05_diffusion_image_generation.png
└── 06_gpt_rl_text_generation.png

train_cnn.py
train_diffusion.py
train_energy.py
train_llm.py
train_mnist_gan.py
train_rl.py
rl_training_rewards.png

Dockerfile
README.md
```

---

# Docker Deployment

## Build

```bash
docker build -t genai-api .
```

## Run

```bash
docker run -p 8000:80 genai-api
```

The container includes all trained models, including the RL-post-trained GPT-2 model used by the `/generate_with_llm` endpoint.

Once the container is running, open

```
http://localhost:8000/docs
```

to access the interactive Swagger UI.

---

# Technologies

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- torchvision
- spaCy
- NumPy
- Pillow (PIL)
- Docker
- Uvicorn

---

# Machine Learning Models

| Model | Dataset | Purpose |
|--------|---------|----------|
| Bigram Language Model | Custom text corpus | Text generation |
| spaCy Embeddings | Pretrained | Word embeddings |
| CNN | CIFAR-10 | Image classification |
| Wasserstein GAN | MNIST | Digit generation |
| Energy-Based Model | CIFAR-10 | Image generation |
| Diffusion Model | CIFAR-10 | Image generation |
| GPT-2 (SFT + RL Post-Training) | Question-answering dataset | Question answering / text generation |

---

# Running Locally

Install dependencies

```bash
uv sync
```

Train the GPT-2 language model (optional if training from scratch)

```bash
uv run python train_llm.py
uv run python train_rl.py
```

Start the API

```bash
uv run uvicorn app.main:app --reload
```

Open

```
http://localhost:8000/docs
```

to test all endpoints interactively.

---

# Author

Developed by **Mariela Bracete** as part of Columbia University's **Applied Generative AI** graduate coursework.
