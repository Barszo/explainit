# import json, os

# session_id = "dd8d06ed-64d1-4237-a014-6f5003df2394" #"PASTE-SESSION-ID-HERE"
# session_dir = os.path.expanduser("~/.factory/sessions/-Users-bartosz-projects-explainit-explainit_project")
# f = os.path.join(session_dir, f"{session_id}.jsonl")

# conversation = []
# with open(f) as fh:
#     for i, line in enumerate(fh):
#         d = json.loads(line)
#         msg = d.get("message")
#         if not isinstance(msg, dict):
#             continue
#         role = msg.get("role","")
#         if role not in ("user","assistant"):
#             continue
#         content = msg.get("content","")
#         texts = []
#         if isinstance(content, list):
#             for item in content:
#                 if isinstance(item, dict) and item.get("type") == "text":
#                     if not item["text"].strip().startswith("<system-reminder>"):
#                         texts.append(item["text"].strip())
#         elif isinstance(content, str) and not content.strip().startswith("<system-reminder>"):
#             texts.append(content.strip())
#         if texts:
#             conversation.append({
#                 "line": i,
#                 "role": role,
#                 "text": "\n".join(texts)
#             })

# out = f"droid_conversation.json"
# with open(out, "w") as fh:
#     json.dump(conversation, fh, indent=2)

# print(f"Saved {len(conversation)} messages to {out}")


import json, os, glob

session_dir = os.path.expanduser("~/.factory/sessions/-Users-bartosz-projects-explainit-explainit_project")
files = sorted(glob.glob(os.path.join(session_dir, "*.jsonl")), key=os.path.getmtime, reverse=True)[:10]

for f in files:
    sid = os.path.basename(f).replace(".jsonl","")
    title = "Untitled"
    first_user_msg = ""
    last_ts = ""
    msg_count = 0

    with open(f) as fh:
        for i, line in enumerate(fh):
            try:
                d = json.loads(line)
                if i == 0:
                    title = d.get("title", "Untitled")
                ts = d.get("timestamp") or ""
                if ts:
                    last_ts = ts
                msg = d.get("message")
                if isinstance(msg, dict):
                    role = msg.get("role","")
                    if role == "user" and not first_user_msg:
                        content = msg.get("content","")
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text = item["text"]
                                    if text.strip().startswith("<system-reminder>"):
                                        continue
                                    if text.strip():
                                        first_user_msg = text.strip().replace("\n"," ")[:200]
                                        break
                    if role in ("user","assistant"):
                        msg_count += 1
            except:
                pass

    print(f"Session: {sid}")
    print(f"  Title:       {title[:80]}")
    print(f"  Last active: {last_ts}")
    print(f"  Msgs:        {msg_count}")
    print(f"  First user:  {first_user_msg if first_user_msg else '(only system-reminders)'}")
    print()