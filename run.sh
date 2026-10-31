#!/bin/bash
# Build the engine (if needed) and start RealFact on http://127.0.0.1:5050
cd "$(dirname "$0")"
make -s all
exec .venv/bin/python -m backend.app
