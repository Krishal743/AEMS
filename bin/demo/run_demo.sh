#!/usr/bin/env bash
set -e

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Enter your search query (text):"
read -r QUERY

python3 bin/demo/demo.py --query "$QUERY" --top-k 5
