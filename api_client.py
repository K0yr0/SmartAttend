import requests
from typing import List, Dict

# Your n8n webhook URL
N8N_WEBHOOK_URL = "https://adam143-20143.wykr.es/webhook/attendance"


def send_attendance_to_n8n(records: List[Dict]) -> None:
    """
    Sends attendance records to n8n as JSON.

    Expected format of `records`:
    [
      {
        "name": "Alice",
        "emotion": "happy",
        "timestamp": "2025-12-05T15:23:00"
      },
      ...
    ]
    """
    if not records:
        print("[INFO] No records to send to n8n.")
        return

    payload = {"records": records}

    try:
        print(f"[INFO] Sending {len(records)} records to n8n...")
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)

        if resp.status_code >= 200 and resp.status_code < 300:
            print("[OK] n8n accepted the data.")
        else:
            print(f"[WARN] n8n responded with status {resp.status_code}: {resp.text}")

    except Exception as e:
        # Don't crash the whole app just because the network failed
        print(f"[ERROR] Failed to send data to n8n: {e}")
