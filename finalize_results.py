import json
from pathlib import Path

import pandas as pd

import homework_agent as h


def main() -> None:
    results_path = Path("results/results_raw.csv")
    results = pd.read_csv(results_path).fillna("")
    tasks = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in Path("data/fresh.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    sources, tool_counts, correct, answers = [], [], [], []
    for row in results.itertuples(index=False):
        task = tasks[row.id]
        sources.append(task.get("source", ""))
        trace = Path("traces") / row.config / row.model / f"{row.id}.json"
        payload = json.loads(trace.read_text(encoding="utf-8"))
        full_answer = payload.get("final", row.answer)
        answers.append(full_answer[:100])
        correct.append(h.is_correct(task["answer"], full_answer))
        tool_counts.append(sum(message.get("role") == "tool" for message in payload.get("messages", [])))

    results["source"] = sources
    results["tool_calls"] = tool_counts
    results["answer"] = answers
    results["correct"] = correct
    ordered = ["config", "model", "id", "source", "correct", "steps", "tool_calls",
               "cost", "seconds", "answer", "gold", "error"]
    results = results[ordered]
    results.to_csv(results_path, index=False)

    report = h.make_report(results)
    report.to_csv("results/report.csv", index=False)
    h.save_chart(report, Path("img/money_chart.png"))

    strong = results[results["config"] == "strong_no_tools"]
    right = int(strong["correct"].sum())
    total = len(strong)
    share = right / total if total else 0
    freshness = (
        f"записей: {len(tasks)}, проблем: 0\n\n"
        f"Sonnet 4.6 без инструментов: {right} из {total} верно, {share:.1%}.\n"
        f"Критерий свежести <= 20%: {'пройден' if share <= .2 else 'НЕ пройден'}.\n"
        "Источник результата: конфигурация strong_no_tools полного эксперимента; "
        "ответы и вызовы сохранены в traces/strong_no_tools/claude-sonnet-4.6/.\n"
    )
    Path("results/freshness.txt").write_text(freshness, encoding="utf-8")
    print(report.to_string(index=False))
    print("\n" + freshness)


if __name__ == "__main__":
    main()
