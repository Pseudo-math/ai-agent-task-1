from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6",
}
WIKI_HEADERS = {"User-Agent": "agents-course-homework01/1.0 (educational project)"}


@dataclass
class Ledger:
    calls: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, tag: str, model: str, usage: dict, seconds: float) -> None:
        with self.lock:
            self.calls.append({
                "tag": tag,
                "model": model.split("/")[-1],
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "cost": usage.get("cost") or 0.0,
                "seconds": seconds,
            })

    @property
    def total(self) -> float:
        with self.lock:
            return sum(call["cost"] for call in self.calls)


ledger = Ledger()
thread_usage = threading.local()
anthropic_rate_lock = threading.Lock()
last_anthropic_call = 0.0
wiki_rate_lock = threading.Lock()
last_wiki_call = 0.0


def respect_provider_rate_limit(model: str) -> None:
    global last_anthropic_call
    if not model.startswith("anthropic/"):
        return
    with anthropic_rate_lock:
        delay = 3.2 - (time.monotonic() - last_anthropic_call)
        if delay > 0:
            time.sleep(delay)
        last_anthropic_call = time.monotonic()


def post_with_retry(body: dict, attempts: int = 5) -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY не найден")
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "agents-course-homework01",
    }
    problem = "нет ответа"
    for attempt in range(attempts):
        try:
            response = requests.post(CHAT_URL, headers=headers, json=body, timeout=120)
            if response.status_code == 200:
                return response.json()
            problem = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except requests.RequestException as exc:
            problem = f"{type(exc).__name__}: {exc}"
        time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(problem)


def chat(messages: list[dict], model: str, tools: list[dict] | None = None) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "usage": {"include": True}}
    if tools:
        body.update({"tools": tools, "tool_choice": "auto"})
    respect_provider_rate_limit(model)
    started = time.perf_counter()
    data = post_with_retry(body)
    elapsed = time.perf_counter() - started
    usage = data.get("usage") or {}
    ledger.add("agent", model, usage, elapsed)
    thread_usage.cost = getattr(thread_usage, "cost", 0.0) + (usage.get("cost") or 0.0)
    return data["choices"][0]["message"]


def wiki_json(language: str, params: dict) -> dict:
    global last_wiki_call
    url = f"https://{language}.wikipedia.org/w/api.php"
    problem = "нет ответа"
    for attempt in range(5):
        with wiki_rate_lock:
            delay = 0.3 - (time.monotonic() - last_wiki_call)
            if delay > 0:
                time.sleep(delay)
            last_wiki_call = time.monotonic()
        try:
            response = requests.get(url, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("info", "Wikipedia API error"))
                return data
            problem = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except requests.RequestException as exc:
            problem = f"{type(exc).__name__}: {exc}"
        time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(problem)


class SearchArgs(BaseModel):
    query: str = Field(description="Короткий поисковый запрос на языке вопроса")
    language: str = Field(default="en", description="Код Wikipedia: en или ru")


class PageFindArgs(BaseModel):
    title: str = Field(description="Точное название найденной статьи Wikipedia")
    keywords: str = Field(description="Два-пять ключевых слов из вопроса")
    language: str = Field(default="en", description="Код Wikipedia: en или ru")


class CalcArgs(BaseModel):
    expr: str = Field(description="Арифметическое выражение")


class ExecArgs(BaseModel):
    code: str = Field(description="Код Python; результат нужно вывести через print")


def web_search(query: str, language: str = "en") -> str:
    try:
        hits = wiki_json(language, {"action": "query", "list": "search", "srsearch": query, "srlimit": 3})["query"]["search"]
        if not hits:
            return "No results"
        return "\n".join(f"[{hit['title']}] {BeautifulSoup(hit.get('snippet', ''), 'html.parser').get_text(' ', strip=True)}" for hit in hits)
    except Exception as exc:
        return f"Search error: {type(exc).__name__}: {exc}"


def page_find(title: str, keywords: str, language: str = "en") -> str:
    try:
        data = wiki_json(language, {"action": "parse", "page": title, "prop": "text"})
        html = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "html.parser")
        segments = []
        for node in soup.select("td.description, p, li, tr"):
            text = " ".join(node.get_text(" ", strip=True).split())
            if 30 <= len(text) <= 4000:
                segments.append(text)
        words = [word for word in re.findall(r"\w+", keywords.casefold()) if len(word) > 2]
        if not words:
            return "No keywords"
        ranked = []
        for segment in dict.fromkeys(segments):
            score = sum(word in segment.casefold() for word in words)
            if score:
                ranked.append((score, segment))
        if not ranked:
            return "Keywords not found on the page"
        ranked.sort(key=lambda item: (-item[0], len(item[1])))
        return "\n".join(text for _, text in ranked[:6])[:7000]
    except Exception as exc:
        return f"Page error: {type(exc).__name__}: {exc}"


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.USub: operator.neg}


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("разрешены только числа и арифметические операции")


def calculator(expr: str) -> str:
    try:
        if not expr.strip():
            return "Calculation error: empty expression"
        value = _evaluate(ast.parse(expr.replace(",", "."), mode="eval").body)
        return str(int(value)) if float(value).is_integer() else str(value)
    except Exception as exc:
        return f"Calculation error: {type(exc).__name__}: {exc}"


def python_exec(code: str) -> str:
    if not code.strip():
        return "Python result is empty"
    try:
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=5)
        output = (result.stdout + result.stderr).strip()
        return output[:5000] if output else "Python result is empty"
    except subprocess.TimeoutExpired:
        return "Python error: timeout"
    except Exception as exc:
        return f"Python error: {type(exc).__name__}: {exc}"


TOOLS = {}


def register(fn, args_model, description: str) -> None:
    TOOLS[fn.__name__] = {
        "fn": fn,
        "args": args_model,
        "schema": {"type": "function", "function": {
            "name": fn.__name__, "description": description,
            "parameters": args_model.model_json_schema(),
        }},
    }


register(web_search, SearchArgs, "Ищет статьи Wikipedia. Сначала найди точное название статьи.")
register(page_find, PageFindArgs, "Читает полный текст и HTML-таблицы статьи и возвращает фрагменты по ключевым словам.")
register(calculator, CalcArgs, "Безопасно вычисляет арифметическое выражение.")
register(python_exec, ExecArgs, "Запускает небольшой код Python в изолированном процессе.")

SYSTEM = ("Answer factual questions using tools instead of guessing. First use web_search, then page_find on the best article. "
          "Do not repeat an identical tool call. End with exactly one line FINAL: <short answer>.")


@dataclass
class Run:
    answer: str
    steps: int
    messages: list[dict]
    cost: float
    seconds: float


def run_tool(call: dict) -> dict:
    name = call["function"]["name"]
    try:
        spec = TOOLS[name]
        args = spec["args"].model_validate_json(call["function"]["arguments"])
        result = spec["fn"](**args.model_dump())
    except Exception as exc:
        result = f"Tool error: {type(exc).__name__}: {exc}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:7000]}


def agent(question: str, model: str, tool_names: list[str], max_steps: int = 7) -> Run:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    schemas = [TOOLS[name]["schema"] for name in tool_names]
    seen = set()
    thread_usage.cost = 0.0
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        message = chat(messages, model, schemas or None)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return Run(message.get("content") or "", step, messages, thread_usage.cost, time.perf_counter() - started)
        keys = [(call["function"]["name"], call["function"]["arguments"]) for call in calls]
        if step == max_steps or any(key in seen for key in keys):
            messages.pop()
            messages.append({"role": "user", "content": "Tools are unavailable now. Answer from gathered evidence. End with FINAL: <short answer>."})
            final = chat(messages, model)
            messages.append(final)
            return Run(final.get("content") or "", step + 1, messages, thread_usage.cost, time.perf_counter() - started)
        seen.update(keys)
        messages.extend(run_tool(call) for call in calls)
    raise RuntimeError("agent loop ended unexpectedly")


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text).casefold().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


MONTHS = "january february march april may june july august september october november december".split()


def _date_key(text: str) -> tuple[int, int] | None:
    names = "|".join(MONTHS)
    day_first = re.search(rf"\b(\d{{1,2}})\s+({names})\b", str(text), re.I)
    month_first = re.search(rf"\b({names})\s+(\d{{1,2}})\b", str(text), re.I)
    if day_first:
        return MONTHS.index(day_first.group(2).casefold()) + 1, int(day_first.group(1))
    if month_first:
        return MONTHS.index(month_first.group(1).casefold()) + 1, int(month_first.group(2))
    return None


def is_correct(gold: str, answer: str) -> bool:
    if normalize(gold) in normalize(answer):
        return True
    gold_date, answer_date = _date_key(gold), _date_key(answer)
    return gold_date is not None and gold_date == answer_date


def final_answer(text: str) -> str:
    match = re.search(r"FINAL:\s*(.+)", text or "", re.I)
    return match.group(1).strip() if match else (text or "").strip()


def run_one(task: dict, model: str, tools: list[str], config: str, trace_root: Path) -> dict:
    try:
        result = agent(task["question"], model, tools)
        answer = final_answer(result.answer)
        correct = is_correct(task["answer"], answer)
        error = ""
    except Exception as exc:
        result = Run("", 0, [], 0.0, 0.0)
        answer, correct, error = "", False, f"{type(exc).__name__}: {exc}"
    trace_dir = trace_root / re.sub(r"[^\w.-]+", "_", config) / model.split("/")[-1]
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "answer": result.answer, "final": answer, "correct": correct,
               "cost": result.cost, "seconds": result.seconds, "error": error, "messages": result.messages}
    (trace_dir / f"{task['id']}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"config": config, "model": model.split("/")[-1], "id": task["id"], "correct": correct,
            "source": task.get("source", ""), "steps": result.steps,
            "tool_calls": sum(message.get("role") == "tool" for message in result.messages),
            "cost": result.cost, "seconds": result.seconds,
            "answer": answer[:100], "gold": task["answer"], "error": error}


def run_config(tasks: list[dict], model: str, tools: list[str], config: str, workers: int, trace_root: Path) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task, model, tools, config, trace_root) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 10 == 0 or index == len(tasks):
                print(f"{config}: {index}/{len(tasks)}", flush=True)
    return rows


def make_report(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, model), group in results.groupby(["config", "model"], sort=False):
        correct = int(group["correct"].sum())
        cost = group["cost"].sum()
        rows.append({"config": config, "model": model, "n": len(group),
                     "accuracy": correct / len(group), "cost_per_task": cost / len(group),
                     "cost_per_correct": cost / correct if correct else float("inf"),
                     "avg_steps": group["steps"].mean(), "avg_seconds": group["seconds"].mean()})
    return pd.DataFrame(rows)


def save_chart(report: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = report["cost_per_task"].clip(lower=1e-7) * 100
    ax.scatter(x, report["accuracy"] * 100, s=100, color="#5436A3")
    for px, (_, row) in zip(x, report.iterrows()):
        ax.annotate(f"{row['config']}\n{row['model']}", (px, row["accuracy"] * 100), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlim(x.min() / 1.4, x.max() * 4.5)
    ax.set_ylim(-2, max(55, report["accuracy"].max() * 100 + 8))
    ax.set_xlabel("цена задачи, центы (логарифмическая шкала)")
    ax.set_ylabel("доля верных ответов, %")
    ax.grid(alpha=.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fresh.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = весь набор")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    configs = [
        ("strong_no_tools", MODELS["strong"], []),
        ("cheap_no_tools", MODELS["cheap"], []),
        ("cheap_search", MODELS["cheap"], ["web_search"]),
        ("cheap_search_page", MODELS["cheap"], ["web_search", "page_find"]),
        ("mid_search_page", MODELS["mid"], ["web_search", "page_find"]),
    ]
    trace_root = Path("traces")
    all_rows = []
    for config, model, tools in configs:
        all_rows.extend(run_config(tasks, model, tools, config, args.workers, trace_root))
    results = pd.DataFrame(all_rows)
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/results_raw.csv", index=False)
    report = make_report(results)
    report.to_csv("results/report.csv", index=False)
    save_chart(report, Path("img/money_chart.png"))
    print(report.to_string(index=False))
    print(f"Total OpenRouter cost: ${ledger.total:.4f}")


if __name__ == "__main__":
    main()
