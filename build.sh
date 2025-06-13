#!/usr/bin/env bash
# exit on error
set -o errexit

# Install setuptools and wheel first
pip install --upgrade pip
pip install setuptools==68.2.2 wheel==0.41.3

# Install the rest of the requirements
pip install -r requirements.txt 