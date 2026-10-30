"""The example classifiers shipped in ``classifiers/``."""

from __future__ import annotations

import pytest

from chronominer.common.classifiers import load_classifier, resolve_classifier
from chronominer.output.entries import entry_fields
from chronominer.spec import CLASSIFIERS_DIR, classifier_names, resolve_schema
from tests.test_example_schemas import EXPECTED

SHIPPED = {
    "chunk_kind": "chunk",
    "language": "chunk",
    "document_type": "document",
    "commodity_group": "entry",
    "occupation_sector": "entry",
    "cargo_class": "entry",
    "goods_category": "entry",
    "subject_class": "entry",
    "letter_topic": "entry",
}


def test_the_folder_holds_exactly_the_shipped_classifiers() -> None:
    assert sorted(classifier_names()) == sorted(SHIPPED)
    assert sorted(path.stem for path in CLASSIFIERS_DIR.glob("*.json")) == sorted(
        SHIPPED
    )


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_every_shipped_classifier_loads_with_its_unit(name: str) -> None:
    classifier = load_classifier(resolve_classifier(name, CLASSIFIERS_DIR))
    assert (classifier.name, classifier.unit) == (name, SHIPPED[name])
    assert classifier.description.strip()


@pytest.mark.parametrize(
    "name", sorted(name for name, unit in SHIPPED.items() if unit == "entry")
)
def test_an_entry_classifier_sends_fields_of_a_public_example_schema(
    name: str,
) -> None:
    classifier = load_classifier(CLASSIFIERS_DIR / f"{name}.json")
    assert classifier.schema in EXPECTED.values()
    schema = resolve_schema(classifier.schema or "")
    known = {field.name for field in entry_fields(schema.schema)}
    assert set(classifier.fields or ()) <= known
