import json
import statistics

gate_total = 0
gate_passed = 0
latencies = []

with open("/workspaces/video-intel/data/logs/edge_a.jsonl") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("stage") == "motion_gate":
                gate_total += 1
                if data.get("passed"):
                    gate_passed += 1
            elif data.get("stage") == "detect":
                if "latency_ms" in data:
                    latencies.append(data["latency_ms"])
        except:
            pass

rejected = gate_total - gate_passed
rejection_rate = (rejected / gate_total * 100) if gate_total > 0 else 0
avg_latency = statistics.mean(latencies) if latencies else 0

print("\n=== SYSTEM PERFORMANCE METRICS ===")
print(f"Total Frames Evaluated by Gate: {gate_total}")
print(f"Frames Rejected (CPU Saved!):   {rejected} ({rejection_rate:.1f}%)")
print(f"Average ONNX Inference Latency: {avg_latency:.1f} ms")
print("==================================\n")
