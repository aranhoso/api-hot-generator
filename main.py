from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import re


class GenerationRequest(BaseModel):
    name: str
    temperature: Optional[float] = 0.8
    max_length: Optional[int] = 512


class GenerationResponse(BaseModel):
    prompt: str


app = FastAPI(title="HotWheels Model API")
model_path = "model_complete.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HotWheelsLanguageModel(nn.Module):
    def __init__(self, context_length, vocab_size, embedding_dim=32, hidden_dim=128):
        super().__init__()
        self.context_lenght = context_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = nn.Linear(context_length * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embeddings = self.word_embeddings(x)

        batch_size = x.shape[0]
        x = embeddings.view(batch_size, -1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class HotWheelsTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_tokens = {
            "PAD": "<|pad|>",
            "NAME": "<name>",
            "YEAR": "<year>",
            "COLOR": "<color>",
            "SERIES": "<series>",
            "END": "<|end|>",
        }

    def _add_token(self, token):
        if token not in self.char_to_id:
            idx = len(self.char_to_id)
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

    def _tokenize(self, text):
        pattern = r"""<\|pad\|>|<name>|<year>|<color>|<series>|<\|end\|>|."""
        return re.findall(pattern, text)

    def encode(self, text):
        tokens = self._tokenize(text)
        return [
            self.char_to_id.get(token, self.char_to_id[self.special_tokens["PAD"]])
            for token in tokens
        ]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, self.special_tokens["PAD"]) for i in ids)


def clean_text(text):
    text = re.sub(r"<\|pad\|>|<\|end\|>|<name>|<year>|<color>|<series>", "", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r'[^\w\s\-\'",]', "", text)
    return text.strip()


def extract_car_info(text):
    pattern = r"<name>(.*?)<year>(.*?)<color>(.*?)<series>(.*?)(?:<\|end\|>|$)"
    match = re.search(pattern, text)

    if match:
        return {
            "name": clean_text(match.group(1)),
            "year": clean_text(match.group(2)),
            "color": clean_text(match.group(3)),
            "series": clean_text(match.group(4)),
        }
    return None


def create_image_prompt(car_info):
    if not car_info:
        return None

    if not all(car_info.values()):
        return None

    prompt = (
        f"Generate a HotWheels styled miniature car called {car_info['name']} "
        f"from {car_info['year']}, packaged, from the {car_info['series']} series, "
        f"{car_info['color']} color, centered on a white background, whole case in frame."
    )

    prompt = re.sub(r"\s+", " ", prompt)
    return prompt.strip()


def sample_with_temperature(logits, temperature):
    if temperature < 1e-6:
        return torch.argmax(logits).item()

    probs = F.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


@app.on_event("startup")
async def load_model():
    global model, tokenizer, context_length

    try:
        checkpoint = torch.load(model_path, map_location=device)

        tokenizer = HotWheelsTokenizer()
        tokenizer.char_to_id = checkpoint["tokenizer_state"]["char_to_id"]
        tokenizer.id_to_char = checkpoint["tokenizer_state"]["id_to_char"]
        tokenizer.special_tokens = checkpoint["tokenizer_state"]["special_tokens"]

        real_embedding_dim = checkpoint["model_state_dict"][
            "word_embeddings.weight"
        ].shape[1]
        print(f"Dimensão real do embedding no checkpoint: {real_embedding_dim}")

        config = checkpoint["config"]
        config["embedding_dim"] = real_embedding_dim
        context_length = config["context_length"]

        model = HotWheelsLanguageModel(
            context_length=config["context_length"],
            vocab_size=config["vocab_size"],
            embedding_dim=real_embedding_dim,
            hidden_dim=config["hidden_dim"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        print(f"Modelo carregado com sucesso no dispositivo: {device}")
        print(f"Dimensões do modelo:")
        print(f"- Vocab size: {config['vocab_size']}")
        print(f"- Embedding dim: {real_embedding_dim}")
        print(f"- Hidden dim: {config['hidden_dim']}")
        print(f"- Context length: {config['context_length']}")

    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        if "checkpoint" in locals():
            print("Conteúdo do checkpoint:", checkpoint.keys())
            if "model_state_dict" in checkpoint:
                for key, tensor in checkpoint["model_state_dict"].items():
                    print(f"{key}: {tensor.shape}")
        raise Exception(f"Falha ao carregar o modelo: {str(e)}")


@app.post("/generate", response_model=GenerationResponse)
async def generate_car(request: GenerationRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não foi carregado")

    prompt = f"<name>{request.name}"

    context = tokenizer.encode(prompt)[-context_length:]

    while len(context) < context_length:
        context.insert(0, tokenizer.char_to_id["<|pad|>"])

    model.eval()
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        with torch.no_grad():
            context_tensor = torch.tensor(
                context[-context_length:], device=device
            ).unsqueeze(0)
            generated = context.copy()

            for _ in range(request.max_length):
                logits = model(context_tensor)
                next_token = sample_with_temperature(logits[0], request.temperature)
                generated.append(next_token)

                if tokenizer.id_to_char[next_token] == "<|end|>":
                    break

                context_tensor = torch.tensor(
                    generated[-context_length:], device=device
                ).unsqueeze(0)

        generated_text = tokenizer.decode(generated)

        car_info = extract_car_info(generated_text)
        image_prompt = create_image_prompt(car_info)

        if image_prompt:
            return GenerationResponse(prompt=image_prompt)

        attempts += 1

    raise HTTPException(
        status_code=500,
        detail="Não foi possível gerar um prompt válido após várias tentativas",
    )
    