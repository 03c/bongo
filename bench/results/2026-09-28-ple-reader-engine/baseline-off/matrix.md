# bongo baseline benchmark — iq2_xs

Generated: `2026-09-28T12:45:39+00:00`  
Endpoint: `http://127.0.0.1:8080/v1`  
Model: `?`  
Harness: `1.0.0`  
Command: `bench/run.sh --repo-root /home/cchild/.paperclip/instances/default/projects/c1ccf5ff-cc7d-4b68-a23c-def2d29bb0a3/15f9f0b7-0021-4164-ae30-902276def736/bongo/.paperclip/worktrees/BAS-62-improve-speed-architecture --tier iq2_xs --contexts 4096,131072 --repeats 1 --max-tokens 128 --needle-context 131072 --hash-mode none --server-pid 2446184 --out-dir bench/results/2026-09-28-ple-reader-engine/baseline-off`

## FATAL

```
{
  "stage": "GET /v1/models",
  "status": null,
  "body": "URLError: <urlopen error [Errno 111] Connection refused>",
  "message": "endpoint unreachable or not an OpenAI-compatible server"
}
```
