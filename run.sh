#!/usr/bin/env bash
# Starts the API from the project root using the .venv312 environment.
cd "$(dirname "$0")" || exit 1
exec ./.venv312/Scripts/python.exe -m uvicorn app.main:app --reload "$@"
