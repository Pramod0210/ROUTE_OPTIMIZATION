# Route Optimizer

A high-performance route optimization tool with a FastAPI backend, Streamlit web UI, and CLI interface.

---

## Features

- **Intelligent Route Optimization** — Find the most efficient paths across multiple stops
- **FastAPI Backend** — RESTful API for programmatic access and integrations
- **Streamlit Dashboard** — Interactive web UI for visualizing and managing routes
- **CLI Support** — Run optimizations directly from the terminal
- **Python 3.12+** — Built on modern Python with async-ready foundations

---

## Requirements

- Python >= 3.12
- pip or uv

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/route-optimizer.git
cd route-optimizer

# Install in editable mode
pip install -e .
```

### Standard Install

```bash
pip install .
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Add your configuration variables here
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Usage

### CLI

```bash
# Run the route optimizer from the command line
route-optimizer --help
```

### FastAPI Server

```bash
# Start the API server
uvicorn route_optimization.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### Streamlit Web UI

```bash
# Launch the interactive dashboard
streamlit run route_optimization/app.py
```

Open your browser at: `http://localhost:8501`

---

## Project Structure

```
route-optimizer/
├── route_optimization/
│   ├── __init__.py
│   ├── cli.py          # CLI entry point
│   ├── main.py         # FastAPI app
│   ├── app.py          # Streamlit UI
│   └── ...
├── pyproject.toml
├── .env
└── README.md
```

---

## API Reference

### `POST /optimize`

Submit a list of locations and receive an optimized route.

**Request Body:**
```json
{
  "locations": [
    { "id": "A", "lat": 28.6139, "lon": 77.2090 },
    { "id": "B", "lat": 19.0760, "lon": 72.8777 }
  ],
  "start": "A"
}
```

**Response:**
```json
{
  "optimized_route": ["A", "B"],
  "total_distance_km": 1150.2
}
```

> Full interactive docs available via Swagger UI at `/docs` when the server is running.

---

## Development

```bash
# Install dev dependencies (if applicable)
pip install -e ".[dev]"

# Run tests
pytest
```

---
