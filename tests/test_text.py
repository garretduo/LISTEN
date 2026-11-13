"""Tests for listen.text.clean_text.

Every input here is a synthetic string written for the test. No story text from the
Learned Hands data (CC BY-NC-SA) is used or quoted.

Run from the repository root:  PYTHONPATH=src python -m pytest tests/test_text.py -q
"""

import doctest

import pytest

from listen import text as text_module
from listen.text import CLEANING_VERSION, clean_text


def test_cleaning_version_is_one():
    assert CLEANING_VERSION == 1


def test_docstring_examples_run():
    # The docstring example is documentation teammates will copy; it must be true.
    result = doctest.testmod(text_module)
    assert result.attempted > 0
    assert result.failed == 0


# Rule 1: line endings


@pytest.mark.parametrize(
    "raw",
    ["a\r\nb", "a\rb", "a\u2028b", "a\u2029b"],
    ids=["crlf", "lone_cr", "line_separator", "paragraph_separator"],
)
def test_line_endings_become_newline(raw):
    assert clean_text(raw) == "a\nb"


def test_crlf_paragraph_break_is_kept():
    assert clean_text("a\r\n\r\nb") == "a\n\nb"


# Rule 2: HTML entities, unescaped exactly once


def test_entity_is_unescaped_once():
    assert clean_text("&amp;gt;") == "&gt;"


def test_double_escaped_entity_loses_only_one_layer():
    assert clean_text("&amp;amp;gt;") == "&amp;gt;"


def test_single_escaped_entities_are_restored():
    assert clean_text("a &lt; b &amp; c &gt; d") == "a < b & c > d"


def test_no_general_second_unescape_for_other_entities():
    # Only &nbsp; is repaired after the single pass (rule 2b); every other entity keeps
    # the single-pass guarantee, so these stay one layer escaped.
    assert clean_text("&amp;amp;gt;") == "&amp;gt;"
    assert clean_text("&amp;amp;") == "&amp;"
    assert clean_text("&amp;lt;") == "&lt;"


# Rule 2b: a literal "&nbsp;" left by double-escaped non-breaking spaces


def test_double_escaped_nbsp_becomes_space():
    assert clean_text("a&amp;nbsp;b") == "a b"


def test_literal_nbsp_entity_becomes_space():
    assert clean_text("a&nbsp;b") == "a b"


def test_repeated_double_escaped_nbsp_collapses_to_one_space():
    # Rule 2b makes two spaces; rule 6 must then collapse them.
    assert clean_text("a&amp;nbsp;&amp;nbsp;b") == "a b"


def test_double_escaped_nbsp_at_the_edges_is_stripped():
    assert clean_text("&amp;nbsp;a&amp;nbsp;") == "a"


def test_double_escaped_nbsp_ends_a_url():
    # Rule 2b runs before rule 3, so the space stops the URL instead of being swallowed.
    assert clean_text("see http://x.com/a&amp;nbsp;more") == "see more"


def test_nbsp_entity_is_matched_by_exact_text_only():
    assert clean_text("&amp;amp;nbsp;") == "&amp;nbsp;"


# Rule 3: links


def test_markdown_link_keeps_text_and_balanced_parentheses_in_url():
    assert clean_text("[the form](https://x.org/a_(b))") == "the form"


def test_markdown_link_inside_sentence():
    assert clean_text("Fill out [the form](http://x.org/f) today") == (
        "Fill out the form today"
    )


def test_bare_url_with_trailing_period_is_removed():
    out = clean_text("see http://x.com/a.")
    assert "http" not in out
    assert out == "see"


def test_url_wrapped_in_parentheses_keeps_closing_parenthesis():
    assert clean_text("(see http://x.com/a)") == "(see )"


def test_www_url_is_removed():
    assert clean_text("go to www.x.com/p now") == "go to now"


def test_https_url_is_removed():
    assert clean_text("a https://x.org/q?id=1&y=2 b") == "a b"


def test_link_text_that_is_a_url_is_removed_too():
    assert clean_text("[http://x.com](http://x.com)") == ""



@pytest.mark.parametrize(
    "url",
    ["HTTP://X.COM/a", "Www.x.com", "HTTPS://x.org/q?id=1", "Http://x.com", "WWW.X.COM/P"],
)
def test_url_scheme_is_matched_in_any_case(url):
    assert clean_text("a " + url + " b") == "a b"


def test_markdown_link_with_uppercase_scheme_keeps_text():
    assert clean_text("[t](HTTPS://x.org)") == "t"
    assert clean_text("[t](Www.x.org/p_(b))") == "t"


def test_uppercase_url_wrapped_in_parentheses_keeps_closing_parenthesis():
    assert clean_text("(see HTTP://X.COM/a)") == "(see )"

# Rule 4: usernames and mentions


@pytest.mark.parametrize(
    "raw",
    ["hi u/abc ok", "hi /u/abc ok", "hi r/legaladvice ok", "hi /r/legaladvice ok", "hi @bob ok"],
    ids=["u", "slash_u", "r", "slash_r", "at"],
)
def test_names_and_mentions_are_removed(raw):
    assert clean_text(raw) == "hi ok"


def test_name_with_hyphen_is_removed_whole():
    assert clean_text("hi u/a-b_c ok") == "hi ok"


def test_names_are_case_sensitive():
    # Reddit writes u/ and r/ in lowercase; only the URL scheme is case-insensitive.
    assert clean_text("hi U/abc and R/legaladvice") == "hi U/abc and R/legaladvice"


def test_name_at_start_of_text_is_removed():
    assert clean_text("u/abc said so") == "said so"


@pytest.mark.parametrize(
    "raw",
    ["he/she", "and/or", "our/r/x", "a@b.com", "mail me at joe@site.org please"],
    ids=["he_she", "and_or", "our_r_x", "email", "email_in_sentence"],
)
def test_slashed_words_and_emails_are_unchanged(raw):
    assert clean_text(raw) == raw


# Rule 5: character fixes


def test_jurisdiction_tag_is_kept():
    assert clean_text("I rent in [CA] and [NY].") == "I rent in [CA] and [NY]."


def test_curly_single_quotes_become_ascii():
    assert clean_text("don\u2019t \u2018x\u2019 \u201a \u201b") == "don't 'x' ' '"


def test_curly_double_quotes_become_ascii():
    assert clean_text("\u201cx\u201d \u201e \u201f") == '"x" " "'


def test_dashes_become_hyphen():
    assert clean_text("a\u2013b\u2014c\u2012d\u2015e") == "a-b-c-d-e"


def test_ellipsis_becomes_three_dots():
    assert clean_text("wait\u2026 what") == "wait... what"


def test_accent_pound_sign_and_emoji_are_kept():
    raw = "caf\u00e9 \u00a350 \U0001F600"
    assert clean_text(raw) == raw


def test_no_break_space_becomes_space():
    assert clean_text("a\u00a0b") == "a b"


@pytest.mark.parametrize("char", ["\u2003", "\u2009", "\u202f", "\u205f", "\u3000"])
def test_other_unicode_spaces_become_space(char):
    assert clean_text("a" + char + "b") == "a b"


@pytest.mark.parametrize("char", ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"])
def test_zero_width_characters_and_soft_hyphen_are_deleted(char):
    assert clean_text("a" + char + "b") == "ab"


def test_entity_nbsp_becomes_space():
    # &nbsp; only becomes U+00A0 after rule 2, so rule 5 must run after it.
    assert clean_text("a&nbsp;b") == "a b"


# Rule 6: whitespace


def test_runs_of_spaces_and_tabs_collapse():
    assert clean_text("a  \t b") == "a b"


def test_blank_lines_are_capped_at_one():
    assert clean_text("a\n \n\n\nb") == "a\n\nb"


def test_spaces_around_newlines_are_removed():
    assert clean_text("a \n b") == "a\nb"


def test_leading_and_trailing_whitespace_is_stripped():
    assert clean_text(" \n\t a b \n\n ") == "a b"


def test_deletions_leave_no_double_spaces():
    assert clean_text("a u/abc b @bob c") == "a b c"


def test_output_only_contains_spaces_and_newlines_as_whitespace():
    raw = "a\x0b\x0c\x1c\x1d\x1e\x1f\x85b\r\n\tc"
    out = clean_text(raw)
    assert {ch for ch in out if ch.isspace()} <= {" ", "\n"}


def test_empty_and_whitespace_only_input_give_empty_string():
    assert clean_text("") == ""
    assert clean_text(" \r\n\t\u00a0") == ""
    assert clean_text("https://x.org u/abc") == ""


# Things the cleaner must leave alone


def test_case_is_not_changed():
    assert clean_text("Hello WORLD") == "Hello WORLD"


def test_markdown_emphasis_is_kept():
    assert clean_text("**bold** and *it*") == "**bold** and *it*"


def test_phone_number_is_kept():
    assert clean_text("call 555-0100 now") == "call 555-0100 now"


# Input type


@pytest.mark.parametrize("bad", [None, 3, 1.5, float("nan"), b"abc", ["a"]])
def test_non_str_input_raises_type_error(bad):
    with pytest.raises(TypeError):
        clean_text(bad)


# Idempotence on text without entity-like sequences (not guaranteed in general)


def test_second_call_changes_nothing_on_plain_text():
    raw = "Hi\u00a0there, see [form](http://x.org/a)\r\n\r\n\r\nthanks u/abc \u2014 [CA]  ok\u2026"
    once = clean_text(raw)
    assert clean_text(once) == once


def test_second_call_is_not_a_noop_for_entity_like_text():
    # Documents the warning in the clean_text docstring: call it once, on raw text.
    assert clean_text(clean_text("&amp;gt;")) == ">"


def test_removing_a_name_can_expose_a_new_match():
    # Documents the second warning in the clean_text docstring.
    assert clean_text("u/a/u/b") == "/u/b"
    assert clean_text(clean_text("u/a/u/b")) == ""
