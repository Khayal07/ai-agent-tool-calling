import re

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent import TraceableAgent


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.invocations = 0
        self.last_messages = None

    def invoke(self, messages):
        self.invocations += 1
        self.last_messages = messages
        step = self.script.pop(0)
        if callable(step):
            step = step(messages)
        content, tool_calls = step
        return AIMessage(content=content, tool_calls=tool_calls)


def tool_call(name, args, call_id="call_1"):
    return {"name": name, "args": args, "id": call_id}


def _make_agent(monkeypatch, max_iterations=5, verbose=False):
    agent = TraceableAgent(max_iterations=max_iterations, verbose=verbose)
    monkeypatch.setattr(
        agent.tools_by_name["get_current_location"],
        "func",
        lambda city=None: f"Şəhər: {city or 'Baku'} | Enlik: 40.3771 | Uzunluq: 49.8875",
    )
    monkeypatch.setattr(
        agent.tools_by_name["get_weather_by_coordinates"],
        "func",
        lambda latitude, longitude: "26.4°C, Günəşli",
    )
    monkeypatch.setattr(
        agent.tools_by_name["convert_celsius_to_fahrenheit"],
        "func",
        lambda celsius: f"{celsius}°C dərəcə {(celsius * 9 / 5 + 32):.1f}°F dərəcəyə bərabərdir.",
    )
    return agent


def _tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


def test_multi_step_chaining_runs_location_weather_convert(monkeypatch):
    script = []

    def step1(messages):
        return "Paris üçün koordinat lazımdır.", [tool_call("get_current_location", {"city": "Paris"}, "c1")]

    def step2(messages):
        return "Koordinat tapıldı, havanı alıram.", [
            tool_call("get_weather_by_coordinates", {"latitude": 48.8534, "longitude": 2.3488}, "c2")
        ]

    def step3(messages):
        last = _tool_messages(messages)[-1].content
        match = re.search(r"([\d.]+)°C", last)
        return "Temperatur alındı, çevirirəm.", [
            tool_call("convert_celsius_to_fahrenheit", {"celsius": float(match.group(1))}, "c3")
        ]

    def step4(messages):
        return "Parijdə temperatur 79.5°F-dir.", []

    script.extend([step1, step2, step3, step4])
    llm = FakeLLM(script)
    agent = _make_agent(monkeypatch)
    agent.llm_with_tools = llm

    result = agent.run("Parijdə temperatur neçə dərəcədir, Fahrenheit ilə?")

    assert llm.invocations == 4
    assert "79.5°F" in result
    assert len(_tool_messages(llm.last_messages)) == 3
    assert all("[TOOL OUTPUT]" in m.content for m in _tool_messages(llm.last_messages))


def test_direct_answer_without_tools(monkeypatch):
    llm = FakeLLM([("Süni intellekt insan zəkasını təqlid edən sistemdir.", [])])
    agent = _make_agent(monkeypatch)
    agent.llm_with_tools = llm

    result = agent.run("Süni intellekt nədir?")

    assert llm.invocations == 1
    assert "Süni intellekt" in result
    assert _tool_messages(llm.last_messages) == []


def test_tool_error_recovery_continues_and_returns_final_answer(monkeypatch):
    agent = _make_agent(monkeypatch)
    monkeypatch.setattr(
        agent.tools_by_name["get_weather_by_coordinates"],
        "func",
        lambda latitude, longitude: (_ for _ in ()).throw(ConnectionError("API əlçatan deyil")),
    )

    def step1(messages):
        return "Havanı almaq istəyirəm.", [
            tool_call("get_weather_by_coordinates", {"latitude": 40.37, "longitude": 49.89}, "c1")
        ]

    def step2(messages):
        return "Hava məlumatı alına bilmədi, üzr istəyirəm.", []

    llm = FakeLLM([step1, step2])
    agent.llm_with_tools = llm

    result = agent.run("Bakıda hava necədir?")

    assert "üzr" in result.lower() or "Üzr" in result
    assert "[TOOL ERROR]" in _tool_messages(llm.last_messages)[0].content


def test_unknown_tool_name_is_handled_gracefully(monkeypatch):
    def step1(messages):
        return "Naməlum tool çağırıram.", [tool_call("nonexistent_tool", {}, "c1")]

    def step2(messages):
        return "Tool tapılmadı, amma davam edirəm.", []

    llm = FakeLLM([step1, step2])
    agent = _make_agent(monkeypatch)
    agent.llm_with_tools = llm

    result = agent.run("Test sorğusu")

    assert "davam" in result
    assert "[TOOL ERROR]" in _tool_messages(llm.last_messages)[0].content
    assert "mövcud deyil" in _tool_messages(llm.last_messages)[0].content


def test_max_iteration_guardrail_stops_runaway_loop(monkeypatch):
    runaway = []
    for i in range(5):
        runaway.append(
            ("dövr", [tool_call("get_current_location", {}, f"c{i}")])
        )
    llm = FakeLLM(runaway)
    agent = _make_agent(monkeypatch, max_iterations=3)
    agent.llm_with_tools = llm

    result = agent.run("sonsuz sorğu")

    assert llm.invocations == 3
    assert "dayandırıldı" in result
    assert "3" in result


def test_max_iteration_guardrail_uses_default_constant():
    from agent import MAX_ITERATIONS

    assert MAX_ITERATIONS == 5


def test_validation_error_forwarded_as_tool_error(monkeypatch):
    agent = _make_agent(monkeypatch)

    def step1(messages):
        return "Yanlış parametr göndərirəm.", [
            tool_call("get_weather_by_coordinates", {"latitude": 95, "longitude": 2.3}, "c1")
        ]

    def step2(messages):
        return "Parametrlər yanlış idi, düzəltdim.", []

    llm = FakeLLM([step1, step2])
    agent.llm_with_tools = llm

    result = agent.run("Səhv koordinat")

    assert "düzəltdim" in result
    assert "[TOOL ERROR]" in _tool_messages(llm.last_messages)[0].content
