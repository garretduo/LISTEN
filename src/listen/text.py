"""Text cleaning shared by the TF-IDF model, the embedding models and the Streamlit app.

Purpose
    One small, deterministic function, ``clean_text``, that turns the raw text of a
    Learned Hands story (or text a user pastes into the app) into the text those
    consumers see. The TF-IDF model, the averaged-embedding models and the app all
    call this same function, so the models that read text start from identical input
    and differ only in features and classifier. It also keeps the README's promise
    that "we remove usernames and links".

Input
    A ``str`` of raw story text, for example one value of the ``full_text`` column.

Output
    A ``str`` with links, usernames and mentions removed, characters normalised and
    whitespace tidied. Nothing else changes: the function does not lowercase, remove
    stopwords or stem. Each consumer decides those things for itself (the TF-IDF
    vectorizer can lowercase on its own, and a pretrained embedding model may want
    the original casing), and a shared cleaner must not make the choice for all of
    them. Markdown emphasis, e-mail addresses and phone numbers are left alone.

The function is a pure function: no randomness, no global state, no I/O, and the
result does not depend on the locale or on the Python session.

Rules, applied in this order (the order matters, see the reasons)

1. Line endings. ``\\r\\n``, a lone ``\\r``, U+2028 and U+2029 become ``\\n``. Exports
   mix these conventions; one convention lets the later rules treat ``\\n`` as the
   only newline. Without it, ``\\r\\n\\r\\n`` would be flattened to a single space by
   rule 6 and the paragraph break would be lost. (A ``\\r`` that only appears after
   rule 2, from an entity such as ``&#13;``, is treated as plain whitespace by rule 6.)
2. HTML entities. ``html.unescape`` runs exactly once. The data stores ``&``, ``<``
   and ``>`` as entities, once (for example ``&gt;``), so a single pass restores
   them. A second pass would wrongly alter text that really contained ``&amp;gt;``.
   It runs before the link rules because the URL patterns stop at whitespace and
   brackets, and entities such as ``&nbsp;`` only turn into those characters once
   they are unescaped; otherwise ``http://x.com&nbsp;more`` would be swallowed whole.
2b. Leftover ``&nbsp;``. Every literal ``&nbsp;`` is replaced by a single space. The
   corpus double-escapes non-breaking spaces (``&amp;nbsp;``, 75 occurrences in 12 of
   the 3,459 best-guess stories, checked 2026-11-13), so the single unescape of rule 2
   leaves the literal text ``&nbsp;`` behind, and the models would see it as the token
   "nbsp". This is a targeted repair, not a second unescape: every other entity keeps
   the single-pass guarantee. It runs right after rule 2 and before rule 3, so the
   space it produces ends a URL (as the unescaped ``&nbsp;`` of rule 2 does) and is
   collapsed with neighbouring whitespace by rule 6.
   Known limitation: the only other double-escaped entity in the corpus is
   ``&amp;amp;`` (1 story); a single unescape leaves a literal ``&amp;`` there and
   nothing repairs it.
3. Links. A Markdown link ``[text](http://...)`` is replaced by its link text, then
   any remaining bare ``http://``, ``https://`` or ``www.`` URL is deleted. Both
   patterns allow one level of balanced parentheses inside the URL (Wikipedia-style
   addresses), and a closing ``)`` that only wraps a URL is left in place. URLs are
   identifying (the README promise) and are noise for both TF-IDF and embeddings.
   The scheme (``http://``, ``https://`` or ``www.``) is matched in any case, so
   ``HTTP://X.COM`` and ``Www.x.com`` are removed too. The rest of a bare URL is
   whatever follows the scheme up to whitespace or a bracket.
4. Usernames and mentions. ``u/name``, ``/u/name``, ``r/name``, ``/r/name`` and
   ``@name`` are deleted. Lookbehinds stop the rule from firing inside ordinary text
   such as ``he/she``, ``and/or``, ``our/r/x`` or an e-mail address. This runs after
   rule 3 because rule 3 keeps the text of Markdown links, and that text is often a
   name itself (``[r/legaladvice](https://...)``). Names are matched in lowercase
   only (``u`` and ``r``): Reddit writes them that way, so ``U/abc`` is left alone.
5. Character fixes. Zero-width characters and the soft hyphen are deleted. Unicode
   space characters (no-break space, en and em spaces, ideographic space, ...) become
   a plain space. Curly quotes become ``'`` and ``"``, the dash variants become
   ``-``, and the ellipsis character becomes ``...``. This way "don't" typed with a
   curly or a straight apostrophe is one token. Accents, the pound sign, emoji and
   bracketed jurisdiction tags such as ``[CA]`` are kept on purpose, because they
   carry information about the legal issue.
6. Whitespace. Every run of whitespace other than a newline becomes one space, spaces
   around a newline are dropped, three or more newlines become two (one blank line),
   and leading and trailing whitespace is stripped. The deletions in rules 3 to 5
   leave gaps; this rule closes them.

Changing the rules (``CLEANING_VERSION``)
    ``CLEANING_VERSION`` is an integer that identifies the exact rule set above. If
    you change any rule, regular expression or translation table in this file, add 1
    to ``CLEANING_VERSION`` in the same commit. A different rule set produces
    different cleaned text, and so different near-duplicate groups. The frozen
    train/validation/test splits depend on those groups, so after a bump they must
    be regenerated, every result that used the old splits must be re-run, and an
    entry recording the change and the new version must be added to
    ``docs/decisions.md``. Do not bump it for edits that leave the output unchanged
    (comments, docstrings, renaming a variable).
"""

import html
import re

# Identifies the rule set in the module docstring. Bump it on any change that can alter
# the output of clean_text; see "Changing the rules" above for what that implies.
CLEANING_VERSION = 1

# Rule 1. Some exports use \r\n or a bare \r, and U+2028/U+2029 are Unicode line and
# paragraph separators. Without this step they would reach rule 6 as ordinary
# whitespace and blur the paragraph breaks that rule 6 deliberately keeps.
_LINE_ENDINGS = re.compile(r"\r\n?|[\u2028\u2029]")

# Rule 2b. A literal "&nbsp;" left after the single html.unescape of rule 2. The corpus
# double-escapes non-breaking spaces ("&amp;nbsp;"), so one unescape turns them into this
# text instead of a space. A second unescape would repair it but also change text that
# is correctly single-escaped, so only this one entity is handled, by name.
_NBSP_ENTITY = re.compile(r"&nbsp;")

# Rule 3, first pattern. A Markdown link: [text](http://...). Group 1 is the link text,
# which is kept because it is usually ordinary words ("the form"). The URL part allows
# one level of balanced parentheses ("a_(b)") so the link's own closing ")" is found.
# The scheme is matched in any case (the inline (?i:...) covers only the scheme), because
# users paste links such as "HTTPS://X.ORG" into the app.
_MARKDOWN_LINK = re.compile(
    r"\[([^\]]*)\]\(\s*(?i:https?://|www\.)[^\s()]*(?:\([^\s()]*\)[^\s()]*)*\s*\)"
)

# Rule 3, second pattern. A bare URL left after Markdown links are handled. The
# character class excludes ( ) [ ] so that a ")" which merely wraps the URL, as in
# "(see http://x.com)", is not swallowed; the optional group still lets a URL contain
# one balanced pair of parentheses. As in the Markdown pattern, the scheme alone is
# case-insensitive.
_BARE_URL = re.compile(
    r"(?i:https?://|www\.)[^\s()\[\]]*(?:\([^\s()]*\)[^\s()\[\]]*)*"
)

# Rule 4, first pattern. Reddit user and subreddit names (u/name, /u/name, r/name,
# /r/name). The lookbehind refuses a match that follows a word character or "/", so
# "our/r/x" and similar slashed words are not mistaken for a subreddit. Deliberately
# case-sensitive (lowercase u and r only): Reddit writes these names in lowercase, and
# a capital letter is more likely ordinary text than a username.
_REDDIT_NAME = re.compile(r"(?<![\w/])/?[ur]/[\w-]+")

# Rule 4, second pattern. An @mention. The lookbehind refuses a match that follows a
# word character, "@" or ".", which keeps e-mail addresses like a@b.com intact.
_AT_MENTION = re.compile(r"(?<![\w@.])@\w+")

# Rule 5, first pattern. Invisible characters that split a word without showing:
# zero-width space/non-joiner/joiner (U+200B-U+200D), word joiner (U+2060), byte order
# mark (U+FEFF) and soft hyphen (U+00AD). Deleted so the visible word stays one token.
_INVISIBLES = re.compile(r"[\u200b-\u200d\u2060\ufeff\u00ad]")

# Rule 5, second pattern. Unicode space characters that look like a space but are
# different code points: no-break space (U+00A0), U+2000-U+200A, narrow no-break space
# (U+202F), medium mathematical space (U+205F) and ideographic space (U+3000).
_UNICODE_SPACES = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")

# Rule 5, third step. Typographic quotes, dashes and the ellipsis mapped to ASCII, so a
# curly apostrophe and a straight one give the same token. str.translate does this in
# one pass; the ellipsis is the only entry that maps to more than one character.
_ASCII_PUNCTUATION = str.maketrans(
    {
        0x2018: "'",  # left single quotation mark
        0x2019: "'",  # right single quotation mark (the usual curly apostrophe)
        0x201A: "'",  # single low-9 quotation mark
        0x201B: "'",  # single high-reversed-9 quotation mark
        0x201C: '"',  # left double quotation mark
        0x201D: '"',  # right double quotation mark
        0x201E: '"',  # double low-9 quotation mark
        0x201F: '"',  # double high-reversed-9 quotation mark
        0x2013: "-",  # en dash
        0x2014: "-",  # em dash
        0x2012: "-",  # figure dash
        0x2015: "-",  # horizontal bar
        0x2026: "...",  # horizontal ellipsis
    }
)

# Rule 6, first pattern. Any whitespace except a newline (tabs, form feeds, the gaps the
# deletions above leave) collapses to one space. Newlines are excluded on purpose:
# paragraph breaks are the only layout information the cleaner keeps.
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")

# Rule 6, second pattern. Spaces next to a newline would make "a\n \nb" differ from
# "a\n\nb", so they are removed before blank lines are counted.
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")

# Rule 6, third pattern. Cap blank lines at one. Long runs of blank lines carry no
# meaning but would make stories that differ only in them look different.
_EXTRA_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    r"""Remove links, usernames and mentions from raw story text and normalise it.

    Args:
        text (str): Raw story text, exactly as read from the data file (for example one
            value of the ``full_text`` column) or as pasted into the app.

    Returns:
        str: The cleaned text. Always true of it: its only whitespace characters are
        single spaces and newlines (no tabs, carriage returns or Unicode spaces), it
        never has more than two newlines in a row, and it has no leading or trailing
        whitespace. Whatever the patterns of rules 3 and 4 match in the input is
        removed, so ordinary links, ``u/name``, ``r/name`` and ``@name`` do not
        survive; slashed words such as ``he/she`` and e-mail addresses are kept on
        purpose. Link schemes match in any case (``HTTP://X.COM`` is removed), but
        names match in lowercase only (``U/abc`` stays), and links or names split up
        by invisible characters are not recognised, because rule 5 deletes those
        characters only after rules 3 and 4 have run. A double-escaped non-breaking
        space (``&amp;nbsp;``) becomes a space. The result is the empty string if the
        input held nothing but links, names and whitespace.

    Raises:
        TypeError: If ``text`` is not a ``str`` (for example ``None``, ``bytes`` or the
            ``float`` NaN that pandas gives for an empty cell). The fix is to decode
            bytes, and to drop or fill missing values, before calling.

    Example:
        >>> clean_text("I rent in [CA].\r\n\r\n\r\nSee https://example.org/a   for the form.")
        'I rent in [CA].\n\nSee for the form.'

    Note:
        Pass RAW story text and call this function exactly once per story. It is not
        guaranteed to be idempotent: ``"&amp;gt;"`` unescapes to ``"&gt;"`` on the
        first call, and a second call would turn that into ``">"``. Only ``&nbsp;``
        is repaired after that single pass, so ``"&amp;amp;"`` leaves a literal
        ``"&amp;"`` (known limitation, 1 story in the corpus). Removing a name
        can also expose a new match (``"u/a/u/b"`` becomes ``"/u/b"``, which a second
        call empties). For plain text without such cases a second call changes
        nothing, and the tests check one example, but cleaned text should never be
        cleaned again.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects a str, got {type(text).__name__}")

    # Rule 1: one newline convention.
    cleaned = _LINE_ENDINGS.sub("\n", text)

    # Rule 2: unescape exactly once. The data is single-escaped, so a second pass would
    # corrupt text that genuinely contains something like "&amp;".
    cleaned = html.unescape(cleaned)

    # Rule 2b: repair the one double-escaped entity the corpus has in bulk, so no literal
    # "&nbsp;" reaches the models. Before rule 3 so the space ends any URL it follows.
    cleaned = _NBSP_ENTITY.sub(" ", cleaned)

    # Rule 3: Markdown links first (keep the link text), then any bare URL. The order
    # matters: the bare-URL rule would otherwise delete the target but leave "[text]()".
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _BARE_URL.sub("", cleaned)

    # Rule 4: usernames, subreddit names and mentions. After rule 3 because the Markdown
    # link text it keeps is often a name ("[r/legaladvice](https://...)").
    cleaned = _REDDIT_NAME.sub("", cleaned)
    cleaned = _AT_MENTION.sub("", cleaned)

    # Rule 5: invisible characters out, Unicode spaces to " ", typographic punctuation
    # to ASCII. Done after rule 2 because unescaping can produce these characters
    # ("&nbsp;" becomes U+00A0).
    cleaned = _INVISIBLES.sub("", cleaned)
    cleaned = _UNICODE_SPACES.sub(" ", cleaned)
    cleaned = cleaned.translate(_ASCII_PUNCTUATION)

    # Rule 6: close the gaps the deletions left, keep at most one blank line, and strip.
    cleaned = _INLINE_WHITESPACE.sub(" ", cleaned)
    cleaned = _SPACE_AROUND_NEWLINE.sub("\n", cleaned)
    cleaned = _EXTRA_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()
