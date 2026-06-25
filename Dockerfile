FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
