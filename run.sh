#!/usr/bin/env bash

source ./.venv/bin/activate
uvicorn main:app --port ${PORT:-8000}