.PHONY: help install install-dl test lint data ablation demo api ui docker clean

PY := PYTHONPATH=src python3

help:
	@echo "DocAI — common tasks"
	@echo "  make install      Install core (CPU) dependencies"
	@echo "  make install-dl   Install the deep-learning stack (torch/transformers/yolo)"
	@echo "  make data         Generate the synthetic document corpus"
	@echo "  make ablation     Run the OCR preprocessing ablation study"
	@echo "  make demo         Run the full end-to-end evaluation demo"
	@echo "  make test         Run the test suite"
	@echo "  make lint         Run ruff"
	@echo "  make api          Launch the FastAPI inference server"
	@echo "  make ui           Launch the Streamlit operations UI"
	@echo "  make docker       Build the Docker image"

install:
	pip install -r requirements.txt && pip install -e .

install-dl:
	pip install -r requirements-dl.txt

data:
	$(PY) scripts/generate_synthetic_docs.py --n 24 --severity 0.9

ablation:
	$(PY) scripts/run_ocr_ablation.py

demo:
	$(PY) scripts/run_demo.py

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src tests scripts

api:
	$(PY) -m uvicorn docai.api.main:app --reload --port 8000

ui:
	DOCAI_API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py

docker:
	docker build -t docai:latest .

clean:
	rm -rf data/synthetic/*.png data/synthetic/*.txt data/synthetic/*.json results
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
