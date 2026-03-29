import json
import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEXT_MODEL   = os.getenv("OLLAMA_TEXT_MODEL", "llama3")
IMAGE_MODEL  = os.getenv("OLLAMA_IMAGE_MODEL", "llava")

INTENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "intents.json")


def _load_intents() -> list:
    with open(INTENTS_PATH, "r") as f:
        data = json.load(f)
    return [intent["name"] for intent in data["intents"]]


def _clean_json(raw: str) -> str:
    """Extract JSON object from raw LLM output."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    # Grab just the {...} block
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return raw.strip()


def _ollama_chat(prompt: str, model: str, images: list = None) -> str:
    """
    Use /api/chat endpoint (works on ALL Ollama versions).
    Falls back to /api/generate if chat endpoint also fails.
    """
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    payload = {
        "model": model,
        "messages": [message],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 80,
        }
    }

    # Try /api/chat first (all modern Ollama versions)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=180,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "")
        # If 404, fall through to /api/generate
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Ollama server nahi chal raha.\n"
            "Terminal mein run karo: ollama serve\n"
            "Phir models pull karo: ollama pull llama3 && ollama pull llava"
        )

    # Fallback: /api/generate (older Ollama)
    gen_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 300}
    }
    if images:
        gen_payload["images"] = images

    resp2 = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=gen_payload,
        timeout=120,
    )
    resp2.raise_for_status()
    return resp2.json().get("response", "")


def predict(user_input: str) -> dict:
    """Predict intent + entities from text using llama3 via Ollama."""
    try:
        intents = _load_intents()
        intents_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(intents))

        prompt = f"""You are an intent classification and entity extraction system.
        

Available intents:
{intents_list}

Extract relevant entities such as: location, date, time, food_item, quantity, destination.

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.

Example:
{{"intent":"book_flight","confidence":0.95,"entities":{{"destination":"Delhi","date":"tomorrow"}}}}

User input: {user_input}

JSON response:"""

        raw    = _ollama_chat(prompt, TEXT_MODEL)
        result = json.loads(_clean_json(raw))

        return {
            "intent":     result.get("intent", "unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "entities":   result.get("entities", {}),
        }

    except ConnectionError as e:
        return {"intent": "connection_error", "confidence": 0.0,
                "entities": {"error": str(e)}}
    except json.JSONDecodeError as e:
        return {"intent": "parse_error", "confidence": 0.0,
                "entities": {"error": f"JSON parse failed: {e}"}}
    except Exception as e:
        return {"intent": "error", "confidence": 0.0,
                "entities": {"error": str(e)}}


def predict_from_image(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """Extract text/intent from image using llava via Ollama."""
    try:
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        intents   = _load_intents()
        intents_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(intents))

        prompt = f"""You are an intent classification system with vision.

Look at this image. Extract any visible text or understand the visual content.
Classify into one of these intents:
{intents_list}

Extract any entities you see (location, date, food_item, destination, etc.).

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.

Example:
{{"extracted_text":"Book flight to Delhi","intent":"book_flight","confidence":0.92,"entities":{{"destination":"Delhi"}}}}

JSON response:"""

        raw    = _ollama_chat(prompt, IMAGE_MODEL, images=[image_b64])
        result = json.loads(_clean_json(raw))

        return {
            "extracted_text": result.get("extracted_text", ""),
            "intent":         result.get("intent", "unknown"),
            "confidence":     float(result.get("confidence", 0.0)),
            "entities":       result.get("entities", {}),
        }

    except ConnectionError as e:
        return {"extracted_text": "", "intent": "connection_error", "confidence": 0.0,
                "entities": {"error": str(e)}}
    except json.JSONDecodeError as e:
        return {"extracted_text": "", "intent": "parse_error", "confidence": 0.0,
                "entities": {"error": f"JSON parse failed: {e}"}}
    except Exception as e:
        return {"extracted_text": "", "intent": "error", "confidence": 0.0,
                "entities": {"error": str(e)}}
