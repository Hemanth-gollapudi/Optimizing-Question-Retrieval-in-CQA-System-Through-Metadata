#!/bin/bash
# sync_models.sh - Sync trained models from notebooks/outputs to project root outputs

echo "🔄 Syncing trained models from notebooks/outputs to outputs/"
echo "============================================================"

# Create output directories if they don't exist
mkdir -p outputs/checkpoints/bert_category_predictor
mkdir -p outputs

# Check if source files exist
if [ ! -d "notebooks/outputs" ]; then
    echo "❌ notebooks/outputs directory not found"
    echo "   Please run the training notebook first"
    exit 1
fi

# Sync category predictor
if [ -f "notebooks/outputs/checkpoints/bert_category_predictor/model.pt" ]; then
    echo "📦 Copying category predictor model (418MB)..."
    cp notebooks/outputs/checkpoints/bert_category_predictor/model.pt \
       outputs/checkpoints/bert_category_predictor/model.pt
    echo "✅ model.pt copied"
else
    echo "⚠️  model.pt not found in notebooks/outputs"
fi

if [ -f "notebooks/outputs/checkpoints/bert_category_predictor/label_map.json" ]; then
    cp notebooks/outputs/checkpoints/bert_category_predictor/label_map.json \
       outputs/checkpoints/bert_category_predictor/label_map.json
    echo "✅ label_map.json copied"
else
    echo "⚠️  label_map.json not found in notebooks/outputs"
fi

# Sync FAISS index
if [ -f "notebooks/outputs/faiss_index.index" ]; then
    echo "📊 Copying FAISS index..."
    cp notebooks/outputs/faiss_index.index outputs/faiss_index.index
    echo "✅ faiss_index.index copied"
else
    echo "⚠️  faiss_index.index not found in notebooks/outputs"
fi

if [ -f "notebooks/outputs/faiss_index_texts.json" ]; then
    cp notebooks/outputs/faiss_index_texts.json outputs/faiss_index_texts.json
    echo "✅ faiss_index_texts.json copied"
else
    echo "⚠️  faiss_index_texts.json not found in notebooks/outputs"
fi

# Sync PMI dictionaries if they exist
if [ -d "notebooks/outputs/pmi_dicts" ]; then
    echo "📚 Copying PMI dictionaries..."
    mkdir -p outputs/pmi_dicts
    cp -r notebooks/outputs/pmi_dicts/* outputs/pmi_dicts/
    echo "✅ PMI dictionaries copied"
fi

echo ""
echo "============================================================"
echo "✅ Model sync complete!"
echo "============================================================"
echo ""
echo "📋 Synced files:"
ls -lh outputs/checkpoints/bert_category_predictor/ 2>/dev/null
ls -lh outputs/faiss_index* 2>/dev/null
echo ""
echo "🚀 You can now restart the server: python run_server.py"

