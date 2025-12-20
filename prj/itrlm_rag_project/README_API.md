# ITRLM+RAG FastAPI Backend

A multilingual question-answering API powered by ITRLM (Improved Translation-based Language Model) and RAG (Retrieval-Augmented Generation).

## Features

- 🌍 **Language Detection**: Automatically detect the language of input text
- 🔄 **Translation**: Translate text between multiple languages
- 🧹 **Text Processing**: Clean and preprocess text
- 🏷️ **Category Prediction**: Classify questions into categories
- 🤖 **RAG Answer Generation**: Generate answers using retrieval-augmented generation
- 🌐 **Multilingual Query Pipeline**: End-to-end multilingual question answering

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Basic start
python run_server.py

# With auto-reload (development)
python run_server.py --reload

# Custom host and port
python run_server.py --host 0.0.0.0 --port 8080
```

### 3. Access API Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Language Detection

```bash
curl -X POST http://localhost:8000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "¿Dónde puedo comprar billetes de avión baratos?"}'
```

Response:

```json
{
  "detected_language": "es",
  "text": "¿Dónde puedo comprar billetes de avión baratos?"
}
```

### Translation

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "भारत की राजधानी क्या है?",
    "target_lang": "en"
  }'
```

Response:

```json
{
  "translated_text": "What is the capital of India?",
  "source_lang": "hi",
  "target_lang": "en"
}
```

### Text Processing

```bash
curl -X POST http://localhost:8000/process-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Where can I buy CHEAP airline tickets???"}'
```

Response:

```json
{
  "original_text": "Where can I buy CHEAP airline tickets???",
  "processed_text": "where can i buy cheap airline tickets"
}
```

### Category Prediction

```bash
curl -X POST http://localhost:8000/predict-category \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I invest in the stock market?"}'
```

Response:

```json
{
  "category": "Business & Finance",
  "confidence": 0.89
}
```

### Generate Answer (RAG)

```bash
curl -X POST http://localhost:8000/generate-answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Where can I buy cheap airline tickets?"}'
```

Response:

```json
{
  "question": "Where can I buy cheap airline tickets?",
  "answer": "You can buy tickets on airline websites."
}
```

### Multilingual Query (Complete Pipeline)

This is the most powerful endpoint - it handles everything automatically:

```bash
curl -X POST http://localhost:8000/multilingual-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "భారతదేశ రాజధాని నగరం ఏది?",
    "return_english": false
  }'
```

Response:

```json
{
  "original_question": "భారతదేశ రాజధాని నగరం ఏది?",
  "detected_language": "te",
  "question_in_english": "Which is the capital city of India?",
  "answer_in_english": "Delhi",
  "answer_in_original_language": "ఢిల్లీ"
}
```

## Python Client Examples

### Language Detection

```python
import requests

response = requests.post(
    "http://localhost:8000/detect-language",
    json={"text": "テストメッセージです"}
)
print(response.json())
# {'detected_language': 'ja', 'text': 'テストメッセージです'}
```

### Multilingual Query

```python
import requests

response = requests.post(
    "http://localhost:8000/multilingual-query",
    json={
        "question": "¿Cuál es la mejor manera de aprender Python?",
        "return_english": False
    }
)
result = response.json()
print(f"Question: {result['original_question']}")
print(f"Language: {result['detected_language']}")
print(f"Answer: {result['answer_in_original_language']}")
```

## Supported Languages

The API supports translation for 100+ languages including:

- **European**: English, Spanish, French, German, Italian, Portuguese, Russian
- **Asian**: Chinese, Japanese, Korean, Hindi, Telugu, Tamil, Bengali
- **Middle Eastern**: Arabic, Hebrew, Farsi
- And many more...

## Project Structure

```
itrlm_rag_project/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── hmr/
│   ├── lang_pipeline.py     # Language detection & translation
│   ├── text_processing.py   # Text preprocessing
│   ├── category_predictor.py # Category classification
│   └── rag_generator.py     # RAG answer generation
├── data/
│   ├── semeval_loader.py    # SemEval dataset loader
│   └── yahoo_loader.py      # Yahoo Answers loader
├── run_server.py            # Server startup script
├── requirements.txt         # Dependencies
└── README_API.md           # This file
```

## Development

### Running in Development Mode

```bash
python run_server.py --reload
```

This enables auto-reload when you make code changes.

### Testing Endpoints

Use the interactive Swagger UI at http://localhost:8000/docs to test all endpoints with a visual interface.

### Adding New Endpoints

1. Add your endpoint function in `api/main.py`
2. Define request/response models using Pydantic
3. Add appropriate tags for organization
4. Document with clear docstrings

## Performance Tips

- The first request to each endpoint may be slower due to model initialization
- Consider using a model caching strategy for production
- For high-volume deployments, use multiple workers: `--workers 4`
- Enable GPU support by installing `faiss-gpu` instead of `faiss-cpu`

## Troubleshooting

### Models Not Loading

If you see errors about missing models or indices:

1. Ensure you've run the training notebooks
2. Check that the `outputs/` directory contains trained models
3. Verify FAISS index exists at `outputs/faiss_index.index`

### Translation Errors

If translation fails:

1. Check your internet connection (Google Translate API requires internet)
2. Try with a different text
3. Check the detected language is correct

### Port Already in Use

If port 8000 is busy:

```bash
python run_server.py --port 8080
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
