# API de Geração de HotWheels

![Última Atualização](https://img.shields.io/badge/atualizado-2025--02--17-blue)
![Versão Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)

API baseada em Rede Neural com Multi Layer Perceptron para geração de modelos de carros HotWheels e prompts para geração de imagens. Construída com FastAPI e PyTorch.

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/aranhoso/hotwheels-model-api.git
cd hotwheels-model-api
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Inicie o servidor:
```bash
uvicorn main:app --reload
```

2. A API estará disponível em `http://localhost:8000`

## Documentação da API

### Geração de Descrição

**Endpoint**: `/generate`

**Método**: POST

**Corpo da Requisição**:
```json
{
    "name": "string",
    "temperature": float,  // opcional, padrão: 0.8
    "max_length": integer  // opcional, padrão: 512
}
```

**Resposta**:
```json
{
    "prompt": "string"
}
```

**Exemplo de Uso**:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "name": "Skyline",
        "temperature": 0.8,
        "max_length": 512
    }
)

print(response.json()["prompt"])
```

## Requisitos

- Python 3.12+
- FastAPI
- PyTorch
- Pydantic
- uvicorn

Instale os requisitos com:
```bash
pip install -r requirements.txt
```

## Autor

Criado por [@aranhoso](https://github.com/aranhoso)