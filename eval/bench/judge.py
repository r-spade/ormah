"""Dataset-specific correctness judges, with strict verdict parsing."""

from __future__ import annotations

import json
from dataclasses import asdict

from ormah.background.llm_client import extract_json

# Verbatim per-type and abstention templates from the official evaluator:
# https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
# Unlike Mem0's relaxed LongMemEval judge, a subset is not sufficient.
LONGMEMEVAL_RULES = {
    "single-session-user": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-assistant": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "multi-session": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
}
ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

# Mem0's unified, no-evidence LoCoMo J-score rules (intentionally permissive):
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py
LOCOMO_RULES = 'Label the generated answer as CORRECT or WRONG.\n\n## Rules\n\n1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer\'s list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.\n\n2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.\n\n3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer\'s key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.\n\n4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.\n\n5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.\n\n6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.\n\n7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.\n\n## ONLY mark WRONG if:\n- The generated answer contains ZERO correct items from the gold answer\n- The answer addresses a completely different topic\n\n## Question\nQuestion: {question}\nGold answer: {answer}\nGenerated answer: {response}\n\nReturn JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels.'


def judge_prompt(question: dict, answer: str) -> str:
    if question["abstention"] or question["dataset"] == "longmemeval":
        template = (
            ABSTENTION if question["abstention"] else LONGMEMEVAL_RULES[question["question_type"]]
        )
        return template.format(question["question"], question["gold"], answer)
    return LOCOMO_RULES.format(
        question=question["question"], answer=question["gold"], response=answer
    )


def judge_answer(provider, question, answer):
    result = provider.complete(
        judge_prompt(question, answer), item_id=question["question_id"], max_tokens=256
    )
    output = asdict(result)
    if question["dataset"] == "longmemeval" or question["abstention"]:
        label = result.text.strip().lower().rstrip(".")
        if label not in {"yes", "no"}:
            raise ValueError(f"Invalid yes/no judge verdict: {result.text[:200]}")
        output.update(correct=label == "yes", reasoning=result.text)
    else:
        parsed = json.loads(extract_json(result.text))
        if parsed.get("label") not in {"CORRECT", "WRONG"}:
            raise ValueError(f"Invalid LoCoMo judge verdict: {result.text[:200]}")
        output.update(correct=parsed["label"] == "CORRECT", reasoning=parsed.get("reasoning", ""))
    return output
