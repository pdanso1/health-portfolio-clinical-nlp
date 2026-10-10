import os
from types import SimpleNamespace as NS
from unittest.mock import patch

from nlp import claude_parser


def _run(content, stop="end_turn"):
    resp = NS(content=content, stop_reason=stop)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}), \
         patch.object(claude_parser.anthropic, "Anthropic") as cls:
        cls.return_value.messages.create.return_value = resp
        out = claude_parser.parse_note("Pt with T2DM on metformin.")
        return out, cls.return_value.messages.create.call_args.kwargs


def test_haiku55_with_thinking_headroom():
    assert claude_parser.MODEL == "claude-haiku-5-5"
    assert claude_parser.MAX_TOKENS >= 8000


def test_cut_off_json_returns_empty_result_not_partial():
    out, _ = _run([NS(type="text", text='{"conditions": ["T2')], stop="max_tokens")
    assert out == claude_parser.EMPTY_RESULT
