import json
import urllib.request
from pathlib import Path
base = "http://127.0.0.1:8005"
with urllib.request.urlopen(base + "/health", timeout=10) as response:
    assert response.status == 200
with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
    models = json.load(response)
assert any(m["id"] == "Qwen3.5-27B-FP8" for m in models["data"])
payload = {"model": "Qwen3.5-27B-FP8", "messages": [{"role": "user", "content": "Reply in Korean with one short sentence confirming you are running."}], "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)
content = result["choices"][0]["message"]["content"]
assert isinstance(content, str) and content.strip(), result
with Path(__file__).with_name("smoke-test.json").open("x") as file:
    json.dump({"models": models, "request": payload, "response": result}, file, ensure_ascii=False, indent=2)
print(json.dumps({"reply": content, "usage": result.get("usage"), "model": result["model"]}, ensure_ascii=False))
