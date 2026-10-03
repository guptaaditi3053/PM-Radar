#!/usr/bin/env python3
"""PM Radar
Finds public LinkedIn posts about product-management interviews and skills using a
search API (Tavily), summarises them (Gemini free tier, optional), and writes the data
that the dashboard in /docs reads.

No third-party Python packages are needed (standard library only).

Usage (you normally never run this by hand - GitHub Actions does):
    python radar.py                      # normal run using config.json
    python radar.py --days 14            # change the time window
    python radar.py --regions India      # only one region
    python radar.py --query "AI PM interview at Google"   # one custom search
    python radar.py --demo               # write sample data to preview the dashboard
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
DOCS = ROOT / "docs"
UTC = timezone.utc

PM_WORDS = [
    "product manage", "product manager", "product management", "product sense",
    "product strategy", "product roadmap", "roadmap", "prioritization", "prioritisation",
    "pm interview", "apm ", "#pm", "#productmanagement", "#productmanager",
    "product leader", "product thinking", "product case", "product role",
]
TAGS_ALLOWED = ["Interview prep", "PM skills", "AI PM", "Career", "Company-specific",
                "Frameworks", "Case study"]


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- HTTP

class ApiError(Exception):
    def __init__(self, status, detail=""):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def post_json(url, body, headers, timeout=60):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            detail = ""
        raise ApiError(e.code, detail)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ApiError(0, str(e))


# --------------------------------------------------------------------------- URL helpers

def normalize_url(url):
    url = url.split("#")[0].split("?")[0].rstrip("/")
    return re.sub(r"^https?://([a-z0-9-]+\.)?linkedin\.com", "https://www.linkedin.com",
                  url, flags=re.I)


def kind_of(url):
    u = url.lower()
    if "linkedin.com/posts/" in u or "linkedin.com/feed/update/" in u:
        return "post"
    if "linkedin.com/pulse/" in u:
        return "article"
    return None


def date_from_url(url):
    """LinkedIn post ids embed the posting time (first 41 bits = milliseconds)."""
    m = re.search(r"(?:activity|share|ugcPost)[-:](\d{16,20})", url)
    if not m:
        return None
    try:
        ms = int(m.group(1)) >> 22
        d = datetime.fromtimestamp(ms / 1000, UTC)
    except (ValueError, OverflowError, OSError):
        return None
    if datetime(2012, 1, 1, tzinfo=UTC) <= d <= datetime.now(UTC) + timedelta(days=1):
        return d.date().isoformat()
    return None


def author_from_slug(url):
    m = re.search(r"linkedin\.com/posts/([^_/?#]+)_", url)
    if not m:
        return ""
    toks = m.group(1).split("-")
    if len(toks) > 1 and re.search(r"\d", toks[-1]):
        toks = toks[:-1]
    return " ".join(t.capitalize() for t in toks if t)


def parse_title(title, url):
    title = (title or "").strip()
    author, headline = "", title
    m = re.match(r"^(.*?)\s+on\s+LinkedIn\s*:\s*(.*)$", title, re.S | re.I)
    if m:
        author, headline = m.group(1).strip(), m.group(2).strip()
    else:
        # Articles look like "Headline | Author | LinkedIn"
        parts = [p.strip() for p in re.split(r"\s+\|\s+", title) if p.strip()]
        if parts and parts[-1].lower() == "linkedin":
            parts = parts[:-1]
        if len(parts) >= 2:
            headline, author = parts[0], parts[1]
        elif parts:
            headline = parts[0]
    if not author:
        author = author_from_slug(url)
    return author, headline


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- text heuristics

def is_pm_relevant(text):
    t = text.lower()
    return any(w in t for w in PM_WORDS)


def find_companies(text, companies):
    found = []
    for c in companies:
        if re.search(r"(?<![A-Za-z])" + re.escape(c) + r"(?![A-Za-z])", text):
            found.append(c)
    return found


def verify_tag(tag, text, companies_found):
    t = text.lower()
    if tag == "Interview prep":
        return "interview" in t
    if tag == "AI PM":
        return bool(re.search(r"\bai\b|artificial intelligence|\bllm|genai|generative", t))
    if tag == "Company-specific":
        return bool(companies_found)
    return True


def short_summary(text, limit=230):
    text = clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


# --------------------------------------------------------------------------- search (Tavily)

def time_range_for(days):
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


def tavily_search(key, query, days, country, max_results):
    body = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": max(1, min(int(max_results), 20)),
        "include_domains": ["linkedin.com"],
        "time_range": time_range_for(days),
    }
    if country:
        body["country"] = country
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    resp = post_json("https://api.tavily.com/search", body, headers)
    return resp.get("results", []) or []


# --------------------------------------------------------------------------- LLM (Gemini)

def build_prompt(items):
    payload = [{"id": i, "title": it["title"], "text": it["snippet"][:1200]}
               for i, it in enumerate(items)]
    return (
        "You help a product manager track LinkedIn posts. For each item below decide whether "
        "it is genuinely about product management: interviews, skills, career, frameworks, "
        "case studies or AI product management. Job ads, sales pitches and unrelated posts "
        "are NOT relevant.\n"
        "Return ONLY a JSON array with one object per item, in this shape:\n"
        '{"id": <number>, "relevant": true|false, "summary": "<max 35 words, plain English, '
        'what the post says>", "tags": [<zero or more of: '
        + ", ".join(f'"{t}"' for t in TAGS_ALLOWED) +
        '>], "companies": [<company names mentioned>]}\n'
        "Only use the text given; do not invent details.\n\nITEMS:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def parse_llm_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    return data if isinstance(data, list) else []


def llm_enrich(items, models, key, batch_size, dead):
    """Returns {index_in_items: result_dict}. Raises nothing - failures just return less."""
    results = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt = build_prompt(batch)
        done = False
        for model in models:
            if model in dead:
                continue
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": key}
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2,
                                         "responseMimeType": "application/json"}}
            for attempt in range(2):
                try:
                    resp = post_json(url, body, headers, timeout=90)
                    text = resp["candidates"][0]["content"]["parts"][0]["text"]
                    for obj in parse_llm_json(text):
                        idx = obj.get("id")
                        if isinstance(idx, int) and 0 <= idx < len(batch):
                            results[start + idx] = obj
                    done = True
                    break
                except ApiError as e:
                    if e.status == 429 and attempt == 0:
                        log(f"  Gemini ({model}) rate-limited, waiting 30s...")
                        time.sleep(30)
                        continue
                    log(f"  Gemini ({model}) unavailable: {e.status} {e.detail[:120]}")
                    dead.add(model)
                    break
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    log(f"  Could not read Gemini reply ({model}): {e}")
                    break
            if done:
                break
        if not done and all(m in dead for m in models):
            log("  All Gemini models failed - using simple summaries for the rest.")
            return results
        time.sleep(7)  # stay under free-tier requests-per-minute
    return results


# --------------------------------------------------------------------------- demo data

def demo_posts():
    today = datetime.now(UTC).date()

    def d(n):
        return (today - timedelta(days=n)).isoformat()

    rows = [
        ("Asha Verma", "How I prepared for 30+ PM interviews in 3 months",
         "A structured prep plan: product sense on Mondays, execution on Wednesdays, and one mock a week.",
         ["Interview prep", "Frameworks"], ["India"], [], 1),
        ("Rahul Menon", "What 'AI PM' really means at product companies",
         "Breaks AI product management into model quality, data loops, evals and UX guardrails.",
         ["AI PM", "PM skills"], ["Global"], [], 2),
        ("Priya Nair", "Google PM interview: the product sense round decoded",
         "Walks through how candidates are scored on user empathy, structure and prioritisation.",
         ["Interview prep", "Company-specific"], ["Global", "India"], ["Google"], 3),
        ("Karan Shah", "Prioritisation frameworks I actually use as a PM",
         "Compares RICE, ICE and a simple impact-vs-effort grid, and when each one breaks down.",
         ["PM skills", "Frameworks"], ["India"], [], 5),
        ("Meera Iyer", "Swiggy PM case study: reducing delivery-time anxiety",
         "A teardown of a food-delivery problem with metrics, hypotheses and a rollout plan.",
         ["Case study", "Company-specific"], ["India"], ["Swiggy"], 8),
        ("Daniel Brooks", "Eval-driven development for AI products",
         "Why PMs should own the eval set for LLM features and how to start with 50 examples.",
         ["AI PM", "Frameworks"], ["Global"], [], 12),
    ]
    posts = []
    for i, (author, title, summary, tags, regions, companies, ago) in enumerate(rows):
        posts.append({
            "url": f"https://www.linkedin.com/posts/demo-sample-{i + 1}",
            "type": "post", "author": author, "title": title, "snippet": summary,
            "summary": summary, "summary_source": "demo", "date": d(ago),
            "date_source": "post", "first_seen": d(ago), "regions": regions,
            "topics": tags, "tags": tags, "companies": companies, "relevant": True,
            "demo": True,
        })
    return posts


# --------------------------------------------------------------------------- output

def write_output(posts, meta, demo=False):
    visible = [p for p in posts.values() if p.get("relevant") is not False]
    visible.sort(key=lambda p: (p.get("date") or ""), reverse=True)
    payload = {
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "demo": demo,
        "settings": meta,
        "regions": sorted({r for p in visible for r in p.get("regions", [])}),
        "tags": sorted({t for p in visible for t in p.get("tags", [])}),
        "companies": sorted({c for p in visible for c in p.get("companies", [])}),
        "posts": visible,
        "_hidden": [p for p in posts.values() if p.get("relevant") is False],
    }
    # Hidden (irrelevant) posts are remembered in the JSON so they are not re-checked,
    # but the browser-facing JS file only carries the visible ones.
    DOCS.mkdir(exist_ok=True)
    (DOCS / "posts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    public = {k: v for k, v in payload.items() if k != "_hidden"}
    (DOCS / "posts.js").write_text("window.RADAR_DATA = " + json.dumps(public, ensure_ascii=False)
                                   + ";\n", encoding="utf-8")


def load_existing():
    path = DOCS / "posts.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if data.get("demo"):
        return {}
    posts = {}
    for p in data.get("posts", []) + data.get("_hidden", []):
        if p.get("url") and not p.get("demo"):
            posts[p["url"]] = p
    return posts


# --------------------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--days", type=int, help="time window in days (overrides config)")
    ap.add_argument("--regions", help="comma-separated region labels, e.g. 'India,Global'")
    ap.add_argument("--topics", help="comma-separated topic tags to fetch, e.g. 'AI PM'")
    ap.add_argument("--query", help="run ONE custom search instead of the configured topics")
    ap.add_argument("--demo", action="store_true", help="write sample data (no network)")
    ap.add_argument("--no-llm", action="store_true", help="skip Gemini, use simple summaries")
    return ap.parse_args()


def pick_regions(cfg, wanted):
    configured = cfg.get("regions", [])
    if not wanted or wanted.strip().lower() in ("all", "*"):
        return configured
    out = []
    for label in [w.strip() for w in wanted.split(",") if w.strip()]:
        match = next((r for r in configured if r["label"].lower() == label.lower()), None)
        if match:
            out.append(match)
        else:  # unknown label: treat it as a country name
            out.append({"label": label.title(), "country": label.lower(), "keywords": label})
    return out or configured


def main():
    args = parse_args()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if args.demo:
        posts = {p["url"]: p for p in demo_posts()}
        write_output(posts, {"days": 30, "note": "demo data"}, demo=True)
        log(f"Wrote {len(posts)} DEMO posts. Open docs/index.html to preview.")
        return 0

    days = args.days or int(os.getenv("RADAR_DAYS") or cfg.get("time_period_days", 7))
    regions = pick_regions(cfg, args.regions or os.getenv("RADAR_REGIONS", ""))
    custom = args.query or os.getenv("RADAR_QUERY", "").strip()
    only_topics = args.topics or os.getenv("RADAR_TOPICS", "")
    topics = cfg.get("topics", [])
    if custom:
        topics = [{"tag": "Custom", "queries": [custom]}]
    elif only_topics.strip():
        wanted = {t.strip().lower() for t in only_topics.split(",") if t.strip()}
        topics = [t for t in topics if t["tag"].lower() in wanted] or topics

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not tavily_key:
        log("ERROR: TAVILY_API_KEY is missing. Add it under GitHub > Settings > Secrets > Actions.")
        return 1

    max_credits = int(cfg.get("max_credits_per_run", 30))
    max_results = int(cfg.get("max_results_per_query", 20))
    keep_days = int(cfg.get("keep_history_days", 90))
    companies_cfg = cfg.get("companies", [])
    today = datetime.now(UTC).date()
    cutoff = (today - timedelta(days=days)).isoformat()

    posts = load_existing()
    log(f"Window: last {days} days | regions: {[r['label'] for r in regions]} | "
        f"topics: {[t['tag'] for t in topics]}")

    jobs = [(r, t, q) for r in regions for t in topics for q in t["queries"]]
    credits = 0
    new_urls = []
    for region, topic, q in jobs:
        if credits >= max_credits:
            log(f"Reached max_credits_per_run ({max_credits}); stopping search early.")
            break
        query = f"{q} {region.get('keywords', '')}".strip()
        try:
            results = tavily_search(tavily_key, query, days, region.get("country", ""), max_results)
        except ApiError as e:
            if e.status in (401, 403):
                log(f"ERROR: Tavily rejected the API key ({e.status}). Check the TAVILY_API_KEY secret.")
                return 1
            if e.status in (429, 432, 433):
                log(f"Tavily limit reached ({e.status}). Keeping what we have so far.")
                break
            log(f"  Search failed for '{query}': {e}")
            continue
        credits += 1
        kept = 0
        for r in results:
            url = normalize_url(r.get("url", ""))
            kind = kind_of(url)
            if not kind:
                continue
            pdate = date_from_url(url)
            if pdate and pdate < cutoff:
                continue
            rec = posts.get(url)
            if rec is None:
                author, headline = parse_title(r.get("title", ""), url)
                snippet = clean(r.get("content", ""))[:600]
                rec = {"url": url, "type": kind, "author": author,
                       "title": short_summary(headline or snippet, 140), "snippet": snippet,
                       "date": pdate or today.isoformat(),
                       "date_source": "post" if pdate else "first_seen",
                       "first_seen": today.isoformat(), "regions": [], "topics": [],
                       "tags": [], "companies": [], "relevant": None}
                posts[url] = rec
                new_urls.append(url)
            if region["label"] not in rec["regions"]:
                rec["regions"].append(region["label"])
            if topic["tag"] not in rec["topics"]:
                rec["topics"].append(topic["tag"])
            kept += 1
        log(f"  [{region['label']}/{topic['tag']}] '{q}' -> {kept} LinkedIn results")
        time.sleep(0.6)

    # ---- enrich new posts
    fresh = [posts[u] for u in dict.fromkeys(new_urls) if posts[u].get("relevant") is None]
    to_llm = []
    for rec in fresh:
        text = f"{rec['title']} {rec['snippet']}"
        rec["companies"] = find_companies(text, companies_cfg)
        if is_pm_relevant(text):
            rec["relevant"] = True
            to_llm.append(rec)
        else:
            rec["relevant"] = False

    llm_results = {}
    if to_llm and gemini_key and not args.no_llm:
        log(f"Summarising {len(to_llm)} posts with Gemini...")
        llm_cfg = cfg.get("llm", {})
        llm_results = llm_enrich(to_llm, llm_cfg.get("models", ["gemini-2.5-flash-lite"]),
                                 gemini_key, int(llm_cfg.get("batch_size", 15)), set())
    elif to_llm:
        log("No Gemini key (or --no-llm): using snippet-based summaries.")

    for i, rec in enumerate(to_llm):
        res = llm_results.get(i)
        if res:
            if res.get("relevant") is False:
                rec["relevant"] = False
                continue
            rec["summary"] = clean(str(res.get("summary", "")))[:300] or short_summary(rec["snippet"])
            rec["summary_source"] = "ai"
            tags = [t for t in res.get("tags", []) if t in TAGS_ALLOWED]
            for c in res.get("companies", []):
                if isinstance(c, str) and c and c not in rec["companies"] and len(c) < 40:
                    rec["companies"].append(c)
            rec["tags"] = tags
        else:
            rec["summary"] = short_summary(rec["snippet"] or rec["title"])
            rec["summary_source"] = "snippet"
        text = f"{rec['title']} {rec['snippet']}"
        for tag in rec["topics"]:
            if tag in TAGS_ALLOWED and verify_tag(tag, text, rec["companies"]) and tag not in rec["tags"]:
                rec["tags"].append(tag)
        if rec["companies"] and "Company-specific" not in rec["tags"]:
            rec["tags"].append("Company-specific")
        if not rec["tags"]:
            rec["tags"] = ["PM skills"]

    # ---- prune old posts
    keep_cutoff = (today - timedelta(days=keep_days)).isoformat()
    posts = {u: p for u, p in posts.items() if (p.get("date") or "9999") >= keep_cutoff}

    meta = {"days": days, "regions": [r["label"] for r in regions],
            "topics": [t["tag"] for t in topics], "searches_run": credits,
            "new_posts": len([u for u in dict.fromkeys(new_urls)
                              if posts.get(u, {}).get("relevant")])}
    write_output(posts, meta)
    shown = len([p for p in posts.values() if p.get("relevant") is not False])
    log(f"Done. {meta['new_posts']} new relevant posts, {shown} in dashboard, "
        f"{credits} searches used this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
