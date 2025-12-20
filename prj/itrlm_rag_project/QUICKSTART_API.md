# 🚀 FastAPI Backend Quick Start Guide

## What Was Created

A complete FastAPI backend server for the ITRLM+RAG project with the following endpoints:

### 📁 File Structure

```
api/
├── __init__.py           # API module
└── main.py              # FastAPI application (500+ lines)

run_server.py            # Server launcher script
test_api.py             # API testing script
README_API.md           # Complete API documentation
requirements.txt        # Updated with FastAPI dependencies
```

## 🎯 Available Endpoints

| Endpoint               | Method | Purpose                                                             |
| ---------------------- | ------ | ------------------------------------------------------------------- |
| `/`                    | GET    | Root status check                                                   |
| `/health`              | GET    | Health check with component status                                  |
| `/detect-language`     | POST   | Detect language of input text                                       |
| `/translate`           | POST   | Translate text between languages                                    |
| `/process-text`        | POST   | Clean and process text                                              |
| `/predict-category`    | POST   | Predict question category                                           |
| `/generate-answer`     | POST   | Generate RAG-based answer                                           |
| `/multilingual-query`  | POST   | **Complete pipeline: detect → translate → answer → translate back** |
| `/supported-languages` | GET    | List supported languages                                            |

## 🏃 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start the Server

```bash
python run_server.py
```

Or with auto-reload for development:

```bash
python run_server.py --reload
```

### Step 3: Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing the API

### Option 1: Run Test Script

```bash
python test_api.py
```

### Option 2: Use cURL

```bash
# Language Detection
curl -X POST http://localhost:8000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "తెలుగు టెస్ట్"}'

# Multilingual Query (Telugu to Telugu)
curl -X POST http://localhost:8000/multilingual-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "భారతదేశ రాజధాని నగరం ఏది?",
    "return_english": false
  }'
```

### Option 3: Python Client

```python
import requests

# Multilingual Query Example
response = requests.post(
    "http://localhost:8000/multilingual-query",
    json={
        "question": "¿Dónde puedo comprar billetes de avión baratos?",
        "return_english": False
    }
)

result = response.json()
print(f"Original: {result['original_question']}")
print(f"Answer: {result['answer_in_original_language']}")
```

## 🔑 Key Features

### 1. **Language Detection**

Automatically detects 100+ languages using langdetect

### 2. **Translation**

Bidirectional translation via Google Translate API

- Any language → English
- English → Any language

### 3. **Text Processing**

Cleans text using NLTK and regex

- Lowercasing
- Special character removal
- Tokenization

### 4. **Category Prediction**

BERT-based classification into Yahoo Answer categories

- Business & Finance
- Education & Reference
- Entertainment & Music
- And more...

### 5. **RAG Answer Generation**

Retrieval-Augmented Generation using:

- **Retriever**: Sentence-Transformers (all-mpnet-base-v2)
- **Generator**: FLAN-T5-base
- **Index**: FAISS for fast similarity search

### 6. **Multilingual Query Pipeline** (⭐ Main Feature)

End-to-end processing:

1. Detect source language
2. Translate question to English
3. Generate answer using RAG
4. Translate answer back to source language

## 📊 Example Responses

### Multilingual Query (Spanish)

**Request:**

```json
{
  "question": "¿Dónde puedo comprar billetes de avión baratos?",
  "return_english": false
}
```

**Response:**

```json
{
  "original_question": "¿Dónde puedo comprar billetes de avión baratos?",
  "detected_language": "es",
  "question_in_english": "Where can I buy cheap airline tickets?",
  "answer_in_english": "You can buy tickets on airline websites.",
  "answer_in_original_language": "Puede comprar boletos en sitios web de aerolíneas."
}
```

## ⚙️ Configuration

### Server Options

```bash
python run_server.py --help
```

Options:

- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 8000)
- `--reload`: Enable auto-reload (dev mode)
- `--workers`: Number of worker processes (default: 1)

### Environment Setup

The server automatically:

- Loads trained models on startup
- Initializes FAISS index if available
- Sets up translation pipeline
- Validates all components

## 🐛 Troubleshooting

### Issue: Port Already in Use

```bash
python run_server.py --port 8080
```

### Issue: Models Not Found

Run the training notebooks first:

```bash
jupyter notebook notebooks/exploration.ipynb
```

### Issue: FAISS Index Missing

The RAG endpoints need a built index. Check:

- `outputs/faiss_index.index` exists
- `outputs/faiss_index_texts.json` exists

### Issue: Translation Not Working

Requires internet connection for Google Translate API

## 🔐 Production Considerations

1. **API Keys**: Consider using authenticated translation APIs
2. **Rate Limiting**: Add rate limiting for production
3. **Caching**: Cache frequent translations
4. **Monitoring**: Add logging and monitoring
5. **HTTPS**: Use reverse proxy (nginx/caddy) for HTTPS
6. **Workers**: Use multiple workers for scalability

## 📚 Next Steps

1. ✅ Start the server: `python run_server.py`
2. ✅ Test endpoints: `python test_api.py`
3. ✅ Try the interactive docs: http://localhost:8000/docs
4. ✅ Build your frontend/client
5. ✅ Deploy to production

## 🤝 Integration Examples

### React/JavaScript

```javascript
const response = await fetch("http://localhost:8000/multilingual-query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "質問はここに",
    return_english: false,
  }),
});
const data = await response.json();
console.log(data.answer_in_original_language);
```

### Flutter/Dart

```dart
final response = await http.post(
  Uri.parse('http://localhost:8000/multilingual-query'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'question': 'प्रश्न यहाँ है',
    'return_english': false,
  }),
);
final data = jsonDecode(response.body);
print(data['answer_in_original_language']);
```

## 📖 Documentation

- Full API documentation: `README_API.md`
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

**Happy Coding! 🎉**
