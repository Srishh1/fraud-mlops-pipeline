#!/bin/bash
# test_local.sh
# ==============
# Run this BEFORE docker-compose to confirm everything works locally.
#
# Usage: chmod +x test_local.sh && ./test_local.sh

set -e   # exit immediately if any command fails

echo "=============================================="
echo "  WEEK 2 LOCAL VALIDATION SCRIPT"
echo "=============================================="

# ── Step 1: Check model exists ────────────────────────────
echo ""
echo "📋 Step 1: Checking model files..."
if [ ! -f "models/fraud_model_v1.joblib" ]; then
    echo "❌ Model not found! Run: python src/train.py"
    exit 1
fi
if [ ! -f "models/feature_names.json" ]; then
    echo "❌ Feature names not found! Run: python src/train.py"
    exit 1
fi
echo "   ✅ Model files present"

# ── Step 2: Test explainer standalone ─────────────────────
echo ""
echo "🔍 Step 2: Testing SHAP explainer..."
python src/explainer.py
echo "   ✅ Explainer works"

# ── Step 3: Run pytest ────────────────────────────────────
echo ""
echo "🧪 Step 3: Running test suite..."
pytest tests/ -v --tb=short
echo "   ✅ All tests passed"

# ── Step 4: Start API and smoke test ─────────────────────
echo ""
echo "🚀 Step 4: Starting API server (background)..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
sleep 5  # wait for startup

echo "   Testing /health..."
curl -s http://localhost:8000/health | python -m json.tool

echo ""
echo "   Testing /predict with a sample transaction..."
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 500.00,
    "TransactionDT": 10800,
    "card4": 1,
    "card6": 0,
    "D1": 0,
    "D10": 0,
    "dist1": 500.0
  }' | python -m json.tool

echo ""
echo "   Testing /drift-report..."
curl -s http://localhost:8000/drift-report | python -m json.tool

# Cleanup
kill $API_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "  ✅ ALL CHECKS PASSED — Ready for Docker!"
echo "  Next: docker-compose up"
echo "  Then: http://localhost:8000/docs"
echo "=============================================="
