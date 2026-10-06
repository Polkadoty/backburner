#!/usr/bin/env python3
"""api-prefill-bench.py - prefill speed of any OpenAI-compatible server (Splash, llama-server), no restart needed.

Two measurements, both on raw text through /v1/completions with max_tokens=1 and a fresh nonce so no prefix cache hits:

  cold   read N tokens from an empty cache (N in --cold)
  turn   read a --chunk-token tool result on top of a cached D-token context (D in --depths): first D tokens are sent
         once to warm the server's prefix cache, then D + chunk; the second request's time is the read

  bench/api-prefill-bench.py --port 8000 --label splash-mac
Rows: bench/results/api-prefill.jsonl (one per read: kind, depth, tokens, ms, tok/s, label, note).
"""
import argparse, json, os, time, urllib.request, uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8000)
ap.add_argument('--label', required=True, help='what is being measured, e.g. splash-mac or splash-ipad')
ap.add_argument('--note', default='')
ap.add_argument('--cold', default='2048,8192,16384')
ap.add_argument('--depths', default='16384,32768')
ap.add_argument('--chunk', type=int, default=2048)
ap.add_argument('--reps', type=int, default=2)
a = ap.parse_args()
URL = f'http://127.0.0.1:{a.port}'
TEXT = open(os.path.join(ROOT, 'bench', 'prompt.txt')).read()
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)


def get(path):
    return json.loads(urllib.request.urlopen(URL + path, timeout=30).read())


MODEL = get('/v1/models')['data'][0]['id']


def complete(prompt):
    body = dict(model=MODEL, prompt=prompt, max_tokens=1, temperature=0, stream=False)
    t = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(URL + '/v1/completions', json.dumps(body).encode(),
                                                      {'Content-Type': 'application/json'}), timeout=3600)
    out = json.loads(r.read())
    return (time.perf_counter() - t) * 1000, out['usage']['prompt_tokens']


# Characters per token for this text, measured once, so slices land near the token counts asked for.
_, n = complete(TEXT[:40000])
CPT = 40000 / n


def text(tokens, offset=0):
    start = int(offset * CPT) % (len(TEXT) // 2)
    s = (TEXT * 4)[start:start + int(tokens * CPT)]
    return s


def record(**kw):
    os.makedirs(os.path.join(ROOT, 'bench', 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'bench', 'results', 'api-prefill.jsonl'), 'a') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), label=a.label, note=a.note, **kw)) + '\n')


for n in [int(x) for x in a.cold.split(',') if x]:
    for rep in range(a.reps):
        ms, toks = complete(f'[{uuid.uuid4()}]\n' + text(n, rep * 977))
        say(f'cold {toks:6d} tok  {ms / 1000:7.2f} s  {toks / ms * 1000:7.1f} tok/s')
        record(kind='cold', depth=0, tokens=toks, ms=round(ms), tok_s=round(toks / ms * 1000, 1))

for d in [int(x) for x in a.depths.split(',') if x]:
    for rep in range(a.reps):
        ctx = f'[{uuid.uuid4()}]\n' + text(d, rep * 1931)
        _, base = complete(ctx)
        ms, toks = complete(ctx + text(a.chunk, d + 50000 + rep * 313))
        new = toks - base
        say(f'turn depth {base:6d} +{new:5d} tok  {ms / 1000:7.2f} s  {new / ms * 1000:7.1f} tok/s')
        record(kind='turn', depth=base, tokens=new, ms=round(ms), tok_s=round(new / ms * 1000, 1))
