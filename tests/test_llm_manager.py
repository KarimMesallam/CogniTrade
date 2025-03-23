from bot.llm_manager import get_decision_from_llm

def test_get_decision_from_llm():
    prompt = "Test prompt for LLM decision"
    decision = get_decision_from_llm(prompt)
    # For now, your placeholder always returns "BUY"
    assert decision == "BUY"
