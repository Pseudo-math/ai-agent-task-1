import homework_agent as agent


def test_answer_check_accepts_date_order():
    assert agent.is_correct("13 July", "July 13, 2026")
    assert not agent.is_correct("13 July", "July 14, 2026")


def test_calculator_three_cases():
    assert agent.calculator("17*23+5") == "396"
    assert "empty" in agent.calculator("").lower()
    assert "error" in agent.calculator("__import__('os')").lower()


def test_python_exec_three_cases():
    assert agent.python_exec("print(sum(range(10)))") == "45"
    assert "empty" in agent.python_exec("").lower()
    assert "ZeroDivisionError" in agent.python_exec("1/0")


def test_web_search_three_cases(monkeypatch):
    monkeypatch.setattr(agent, "wiki_json", lambda *_: {"query": {"search": [{"title": "Re:Zero season 4", "snippet": "anime season"}]}})
    assert "Re:Zero season 4" in agent.web_search("rezero")
    monkeypatch.setattr(agent, "wiki_json", lambda *_: {"query": {"search": []}})
    assert agent.web_search("missing") == "No results"
    monkeypatch.setattr(agent, "wiki_json", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    assert "error" in agent.web_search("broken").lower()


def test_page_find_three_cases(monkeypatch):
    html = '<table><tr><td class="description">Subaru recruits Meili to cross the Auguria Dunes.</td></tr></table>'
    monkeypatch.setattr(agent, "wiki_json", lambda *_: {"parse": {"text": {"*": html}}})
    assert "Meili" in agent.page_find("Re:Zero season 4", "Meili dunes")
    assert "not found" in agent.page_find("Re:Zero season 4", "Volcanica").lower()
    monkeypatch.setattr(agent, "wiki_json", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    assert "error" in agent.page_find("Broken", "keyword").lower()
