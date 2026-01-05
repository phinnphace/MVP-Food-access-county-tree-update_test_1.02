#!/bin/bash
# Helper script to run the PyShiny app using the local environment
source .new_venv/bin/activate
shiny run --port 8032 dashboard.py
