import json, os

session_id = "dd8d06ed-64d1-4237-a014-6f5003df2394" #"PASTE-SESSION-ID-HERE"
session_dir = os.path.expanduser("~/.factory/sessions/-Users-bartosz-projects-explainit-explainit_project")
f = os.path.join(session_dir, f"{session_id}.jsonl")

conversation = []
with open(f) as fh:
    for i, line in enumerate(fh):
        d = json.loads(line)
        msg = d.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role","")
        if role not in ("user","assistant"):
            continue
        content = msg.get("content","")
        texts = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    if not item["text"].strip().startswith("<system-reminder>"):
                        texts.append(item["text"].strip())
        elif isinstance(content, str) and not content.strip().startswith("<system-reminder>"):
            texts.append(content.strip())
        if texts:
            conversation.append({
                "line": i,
                "role": role,
                "text": "\n".join(texts)
            })

out = f"droid_conversation.json"
with open(out, "w") as fh:
    json.dump(conversation, fh, indent=2)

print(f"Saved {len(conversation)} messages to {out}")