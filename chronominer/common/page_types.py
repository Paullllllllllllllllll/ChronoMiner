"""Page types of printed books and scans, for decision-model classification.

One page image gets one type: its predominant content. The ``general``
profile (the default) fits most printed works; the ``culinary`` profile adds
three types for cookbooks and household books (recipes, menus, carving) that
``general`` counts as ``body_text``. Every stored label carries
:data:`VERSION` and the profile, since a changed description changes the
answers.

The types were chosen on hand-labelled pages of early printed cookbooks
(ChronoDownloader ``docs/jev_eval``): classes a decision model confused in
both directions were merged (contents with index, dedication with preface,
frontispiece with plate, endpaper with blank).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from .decisions import Choice

__all__ = [
    "DEFAULT_PROFILE",
    "NO_TEXT",
    "PROFILES",
    "TITLE_PAGE_THRESHOLD",
    "VERSION",
    "PageType",
    "Profile",
    "no_text_probability",
    "page_type_question",
    "page_types",
    "title_page_index",
]

type Profile = Literal["general", "culinary"]
type Zone = Literal["physical", "front", "body", "back", "other"]

VERSION: Final = "2026-10-29"
"""The taxonomy version stored with every label."""

PROFILES: Final[tuple[Profile, ...]] = ("general", "culinary")
DEFAULT_PROFILE: Final[Profile] = "general"

NO_TEXT: Final = frozenset({"binding", "blank", "scan_target"})
"""Types without text of the book; ``library_mark`` is not one of them."""

TITLE_PAGE_THRESHOLD: Final = 0.5
TITLE_PAGE_SEARCH: Final = 10
""":func:`title_page_index` looks at this many leading pages."""


@dataclass(frozen=True)
class PageType:
    """One page type: its value, its zone in the book and its description."""

    value: str
    zone: Zone
    description: str


_COMMON: Final[tuple[PageType, ...]] = (
    PageType(
        "binding",
        "physical",
        "The outside of the book photographed as an object: cover, boards, spine "
        "or clasp, without printed text of the book. A printed paper wrapper "
        "counts by its text (title page, advertisement).",
    ),
    PageType(
        "blank",
        "physical",
        "An empty leaf: a blank page of the book, a flyleaf, or a plain or marbled "
        "endpaper or pastedown; faint show-through still counts as blank.",
    ),
    PageType(
        "scan_target",
        "physical",
        "Material added by the digitizing library: colour chart, grey scale, "
        "ruler, target card, digitization or copyright slip.",
    ),
    PageType(
        "library_mark",
        "physical",
        "A page whose main content is a bookplate, library stamp, shelfmark "
        "label, barcode, accession number or ownership inscription.",
    ),
    PageType(
        "handwritten",
        "physical",
        "Handwriting fills most of the page: manuscript notes, inscriptions or a "
        "manuscript leaf.",
    ),
    PageType(
        "half_title",
        "front",
        "A half-title: only the short title of the book (no author, no imprint), "
        "before the title page.",
    ),
    PageType(
        "title_page",
        "front",
        "The book's title page: full title, usually author and imprint (place, "
        "printer, year); engraved title pages included.",
    ),
    PageType(
        "illustration",
        "body",
        "A full-page illustration, portrait, engraving, map or folding plate, "
        "including the frontispiece facing the title page.",
    ),
    PageType(
        "preface_or_dedication",
        "front",
        "Preface, foreword, introduction, address to the reader, commendatory "
        "verses, or a dedication to a patron.",
    ),
    PageType(
        "privilege_or_approval",
        "front",
        "Royal privilege, approbation, licence, imprimatur or censor's approval, "
        "wherever printed.",
    ),
    PageType(
        "contents_or_index",
        "front",
        "A list of chapters, sections, subjects or names with page numbers: a "
        "table of contents, register or alphabetical index, at the front or the "
        "back.",
    ),
    PageType(
        "section_title",
        "body",
        "A divisional title: a page announcing a part, book or treatise within "
        "the volume, with little or no running text.",
    ),
)

_GENERAL_BODY: Final = PageType(
    "body_text",
    "body",
    "Running text of the work: prose, instructions, recipes, treatise, letters "
    "or verse.",
)

_CULINARY_BODY: Final[tuple[PageType, ...]] = (
    PageType(
        "recipe_text",
        "body",
        "Running text with recipes: instructions to prepare a dish, drink, "
        "preserve, confection, remedy, distillate or household preparation.",
    ),
    PageType(
        "menu_or_bill_of_fare",
        "body",
        "Menus, bills of fare or lists of dishes for meals, seasons or courses, as "
        "lists or table layouts.",
    ),
    PageType(
        "carving_or_table_service",
        "body",
        "Carving instructions, table settings or serving diagrams and their "
        "explanation, including engraved table plans.",
    ),
    PageType(
        "body_text",
        "body",
        "Other running text: treatise, dietetic, medical, agricultural, moral or "
        "household text that gives no preparation instructions.",
    ),
)

_TAIL: Final[tuple[PageType, ...]] = (
    PageType(
        "table_or_chart",
        "body",
        "A page laid out as a table or chart: weights, measures, prices, "
        "calendars, statistics.",
    ),
    PageType("errata", "back", "A list of errata or corrections."),
    PageType(
        "advertisement",
        "back",
        "Publisher's or bookseller's catalogue, advertisements or announcements "
        "of other books or goods.",
    ),
    PageType(
        "colophon",
        "back",
        "A colophon or final imprint: printer, place and date at the end, or a "
        "printer's device on its own.",
    ),
    PageType(
        "other",
        "other",
        "Anything else (bibliography, printed music, pasted-in clipping), or a "
        "page too damaged or cropped to classify.",
    ),
)

_BY_PROFILE: Final[Mapping[Profile, tuple[PageType, ...]]] = MappingProxyType(
    {
        "general": (*_COMMON, _GENERAL_BODY, *_TAIL),
        "culinary": (*_COMMON, *_CULINARY_BODY, *_TAIL),
    }
)


def page_types(profile: Profile = DEFAULT_PROFILE) -> tuple[PageType, ...]:
    """Return the page types of *profile* in book order.

    Raises:
        ValueError: *profile* is unknown.
    """
    try:
        return _BY_PROFILE[profile]
    except KeyError:
        raise ValueError(f"unknown page-type profile {profile!r}") from None


def page_type_question(
    name: str, subject: str, profile: Profile = DEFAULT_PROFILE
) -> Choice:
    """Return the choice question asking what kind of page *subject* is.

    *subject* names the image in the shared input, such as ``image 3``.
    """
    return Choice(
        name,
        f"What kind of page is {subject}?",
        {t.value: t.description for t in page_types(profile)},
    )


def no_text_probability(probabilities: Mapping[str, float]) -> float:
    """Return the probability that a page carries no text of the book."""
    return sum(probabilities.get(value, 0.0) for value in NO_TEXT)


def title_page_index(
    pages: Sequence[Mapping[str, float]],
    *,
    threshold: float = TITLE_PAGE_THRESHOLD,
    search: int = TITLE_PAGE_SEARCH,
) -> int | None:
    """Return the index of the title page among the leading pages, or None.

    *pages* holds the page-type probabilities of the pages in book order.
    The page with the highest ``title_page`` probability among the first
    *search* pages is the title page if that probability reaches
    *threshold*; otherwise the work shows no title page there.
    """
    leading = [p.get("title_page", 0.0) for p in pages[:search]]
    if not leading:
        return None
    best = max(range(len(leading)), key=leading.__getitem__)
    return best if leading[best] >= threshold else None
