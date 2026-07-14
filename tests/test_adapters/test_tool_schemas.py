"""Focused tests for MCP-exposed tool schemas."""

from __future__ import annotations

from ormah.adapters.tool_schemas import ADMIN_TOOLS, TOOLS


def test_recall_node_exposed_via_mcp_tools():
    recall_node = next((tool for tool in TOOLS if tool["name"] == "recall_node"), None)
    assert recall_node is not None
    assert recall_node["parameters"]["required"] == ["node_id"]


def test_recall_node_not_left_in_admin_only_tools():
    recall_node = next((tool for tool in ADMIN_TOOLS if tool["name"] == "recall_node"), None)
    assert recall_node is None


def test_submit_feedback_schema_advertises_optional_whisper_log_id():
    submit_feedback = next((tool for tool in TOOLS if tool["name"] == "submit_feedback"), None)
    assert submit_feedback is not None
    properties = submit_feedback["parameters"]["properties"]
    assert properties["whisper_log_id"]["type"] == "integer"
    assert "whisper_log_id" not in submit_feedback["parameters"]["required"]
    assert "exact whisper or recall event" in submit_feedback["description"]
