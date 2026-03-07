"""Tests for modules/operations/cost_analysis.py."""

import csv
import json
from pathlib import Path

import pytest

from modules.operations.cost_analysis import (
    MODEL_PRICING,
    CostAnalysis,
    FileStats,
    TokenUsage,
    analyze_jsonl_file,
    calculate_cost,
    extract_token_usage_from_record,
    find_jsonl_files,
    normalize_model_name,
    perform_cost_analysis,
    save_analysis_to_csv,
)


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------

class TestNormalizeModelName:
    def test_exact_match(self):
        assert normalize_model_name("gpt-4o") == "gpt-4o"

    def test_date_suffix_stripped(self):
        assert normalize_model_name("gpt-4o-2025-01-15") == "gpt-4o"

    def test_unknown_model_returned_as_is(self):
        assert normalize_model_name("some-unknown-model") == "some-unknown-model"

    def test_anthropic_model(self):
        assert normalize_model_name("claude-3-5-sonnet") == "claude-3-5-sonnet"

    def test_gemini_model_with_date_suffix(self):
        assert normalize_model_name("gemini-2.5-pro-2025-03-01") == "gemini-2.5-pro"

    def test_empty_string(self):
        assert normalize_model_name("") == ""


# ---------------------------------------------------------------------------
# extract_token_usage_from_record
# ---------------------------------------------------------------------------

class TestExtractTokenUsageFromRecord:
    def test_direct_response_data(self):
        record = {
            "response_data": {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.model == "gpt-4o"
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_nested_response_body(self):
        record = {
            "response": {
                "body": {
                    "response_data": {
                        "model": "gpt-4o-mini",
                        "usage": {
                            "input_tokens": 200,
                            "output_tokens": 80,
                            "total_tokens": 280,
                        },
                    }
                }
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.model == "gpt-4o-mini"
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 80
        assert usage.total_tokens == 280

    def test_cached_tokens_extraction(self):
        record = {
            "response_data": {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 100,
                    "total_tokens": 600,
                    "input_tokens_details": {"cached_tokens": 300},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.cached_tokens == 300

    def test_reasoning_tokens_extraction(self):
        record = {
            "response_data": {
                "model": "o3",
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 200,
                    "total_tokens": 700,
                    "output_tokens_details": {"reasoning_tokens": 150},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.reasoning_tokens == 150

    def test_total_tokens_calculated_if_missing(self):
        record = {
            "response_data": {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 0,
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.total_tokens == 150

    def test_model_from_request_metadata(self):
        record = {
            "response_data": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            "request_metadata": {"model": "gpt-4o"},
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.model == "gpt-4o"

    def test_model_from_request_metadata_payload(self):
        record = {
            "response_data": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            "request_metadata": {"payload": {"model": "gpt-4o-mini"}},
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.model == "gpt-4o-mini"

    def test_no_usage_data_returns_zero_usage(self):
        """When response_data has no 'usage' key, returns TokenUsage with all zeros."""
        record = {"response_data": {"model": "gpt-4o"}}
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.model == "gpt-4o"
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0

    def test_empty_record_returns_zero_usage(self):
        """Empty record still produces a TokenUsage (all zeros)."""
        usage = extract_token_usage_from_record({})
        assert usage is not None
        assert usage.prompt_tokens == 0
        assert usage.model == ""

    def test_prompt_tokens_details_fallback(self):
        record = {
            "response_data": {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 100,
                    "total_tokens": 600,
                    "prompt_tokens_details": {"cached_tokens": 200},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.cached_tokens == 200

    def test_completion_tokens_details_fallback(self):
        record = {
            "response_data": {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 200,
                    "total_tokens": 700,
                    "completion_tokens_details": {"reasoning_tokens": 100},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.reasoning_tokens == 100


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:
    def test_known_model(self):
        # gpt-4o: (2.50, 1.25, 10.00) per million
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            cached_tokens=0,
            completion_tokens=1_000_000,
            model="gpt-4o",
        )
        assert cost == pytest.approx(2.50 + 10.00, abs=0.001)

    def test_with_cached_tokens(self):
        # gpt-4o: input=2.50, cached=1.25, output=10.00
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            cached_tokens=500_000,
            completion_tokens=0,
            model="gpt-4o",
        )
        # uncached = 500k, cached = 500k
        expected = (500_000 * 2.50 / 1e6) + (500_000 * 1.25 / 1e6)
        assert cost == pytest.approx(expected, abs=0.001)

    def test_with_discount(self):
        cost_full = calculate_cost(1_000_000, 0, 1_000_000, "gpt-4o", discount=0.0)
        cost_half = calculate_cost(1_000_000, 0, 1_000_000, "gpt-4o", discount=0.5)
        assert cost_half == pytest.approx(cost_full * 0.5, abs=0.001)

    def test_unknown_model_returns_zero(self):
        cost = calculate_cost(1000, 0, 1000, "nonexistent-model-xyz")
        assert cost == 0.0

    def test_zero_tokens(self):
        cost = calculate_cost(0, 0, 0, "gpt-4o")
        assert cost == 0.0

    def test_gpt54_cost(self):
        # gpt-5.4: (2.50, 0.25, 15.00) per million
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            cached_tokens=0,
            completion_tokens=1_000_000,
            model="gpt-5.4",
        )
        assert cost == pytest.approx(2.50 + 15.00, abs=0.001)

    def test_gpt54_pro_cost(self):
        # gpt-5.4-pro: (30.00, 0.0, 180.00) per million
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            cached_tokens=0,
            completion_tokens=1_000_000,
            model="gpt-5.4-pro",
        )
        assert cost == pytest.approx(30.00 + 180.00, abs=0.001)

    def test_gpt53_chat_cost(self):
        # gpt-5.3-chat-latest: (1.75, 0.175, 14.00) per million
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            cached_tokens=500_000,
            completion_tokens=1_000_000,
            model="gpt-5.3-chat-latest",
        )
        expected = (500_000 * 1.75 / 1e6) + (500_000 * 0.175 / 1e6) + (1_000_000 * 14.00 / 1e6)
        assert cost == pytest.approx(expected, abs=0.001)

    def test_gpt53_codex_cost(self):
        # gpt-5.3-codex: (1.75, 0.175, 14.00) per million
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            cached_tokens=0,
            completion_tokens=1_000_000,
            model="gpt-5.3-codex",
        )
        assert cost == pytest.approx(1.75 + 14.00, abs=0.001)


# ---------------------------------------------------------------------------
# analyze_jsonl_file
# ---------------------------------------------------------------------------

class TestAnalyzeJsonlFile:
    def test_basic_analysis(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        record = {
            "status": "success",
            "response_data": {
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        }
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1
        assert stats.successful_chunks == 1
        assert stats.failed_chunks == 0
        assert stats.prompt_tokens == 100
        assert stats.completion_tokens == 50
        assert stats.model == "gpt-4o-mini"
        assert stats.cost_standard > 0

    def test_failed_chunk(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        record = {"status": "error", "response_data": {}}
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1
        assert stats.failed_chunks == 1

    def test_skips_batch_tracking(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        tracking = {"batch_tracking": {"batch_id": "abc123"}}
        data = {
            "status": "success",
            "response_data": {
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        }
        lines = json.dumps(tracking) + "\n" + json.dumps(data) + "\n"
        jsonl.write_text(lines, encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1

    def test_empty_lines_skipped(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        record = {
            "status": "success",
            "response_data": {
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        }
        jsonl.write_text("\n" + json.dumps(record) + "\n\n", encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1

    def test_malformed_json_line(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        jsonl.write_text("not-json\n", encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 0

    def test_nonexistent_file(self, tmp_path):
        stats = analyze_jsonl_file(tmp_path / "nonexistent.jsonl")
        assert stats.total_chunks == 0

    def test_success_inferred_from_response_body(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        record = {
            "response": {
                "body": {
                    "response_data": {
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                }
            }
        }
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1
        assert stats.successful_chunks == 1

    def test_multiple_records(self, tmp_path):
        jsonl = tmp_path / "test_temp.jsonl"
        records = []
        for i in range(5):
            records.append(json.dumps({
                "status": "success",
                "response_data": {
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                },
            }))
        jsonl.write_text("\n".join(records) + "\n", encoding="utf-8")

        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 5
        assert stats.prompt_tokens == 500
        assert stats.completion_tokens == 250


# ---------------------------------------------------------------------------
# find_jsonl_files
# ---------------------------------------------------------------------------

class TestFindJsonlFiles:
    def test_input_is_output_mode(self, tmp_path):
        input_dir = tmp_path / "schema_input"
        input_dir.mkdir()
        (input_dir / "file1_temp.jsonl").write_text("{}", encoding="utf-8")
        (input_dir / "file2.json").write_text("{}", encoding="utf-8")  # not matched

        paths_config = {"general": {"input_paths_is_output_path": True}}
        schemas_paths = {"MySchema": {"input": str(input_dir)}}

        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1
        assert result[0].name == "file1_temp.jsonl"

    def test_output_mode(self, tmp_path):
        output_dir = tmp_path / "schema_output"
        output_dir.mkdir()
        (output_dir / "file1_temp.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": False}}
        schemas_paths = {"MySchema": {"output": str(output_dir)}}

        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1

    def test_output_mode_temp_subfolder(self, tmp_path):
        output_dir = tmp_path / "schema_output"
        temp_folder = output_dir / "temp_jsonl"
        temp_folder.mkdir(parents=True)
        (temp_folder / "file1_temp.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": False}}
        schemas_paths = {"MySchema": {"output": str(output_dir)}}

        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1

    def test_nonexistent_directory(self):
        paths_config = {"general": {"input_paths_is_output_path": True}}
        schemas_paths = {"MySchema": {"input": "/nonexistent/path"}}

        result = find_jsonl_files(paths_config, schemas_paths)
        assert result == []

    def test_deduplication(self, tmp_path):
        input_dir = tmp_path / "schema_input"
        input_dir.mkdir()
        (input_dir / "file1_temp.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": True}}
        # Two schemas pointing to same directory
        schemas_paths = {
            "Schema1": {"input": str(input_dir)},
            "Schema2": {"input": str(input_dir)},
        }

        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1

    def test_empty_schemas(self):
        result = find_jsonl_files({"general": {}}, {})
        assert result == []


# ---------------------------------------------------------------------------
# perform_cost_analysis
# ---------------------------------------------------------------------------

class TestPerformCostAnalysis:
    def test_empty_file_list(self):
        analysis = perform_cost_analysis([])
        assert analysis.total_files == 0
        assert analysis.total_chunks == 0
        assert analysis.total_cost_standard == 0.0

    def test_aggregation(self, tmp_path):
        for i in range(3):
            jsonl = tmp_path / f"file{i}_temp.jsonl"
            record = {
                "status": "success",
                "response_data": {
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                },
            }
            jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        files = sorted(tmp_path.glob("*_temp.jsonl"))
        analysis = perform_cost_analysis(files)

        assert analysis.total_files == 3
        assert analysis.total_chunks == 3
        assert analysis.total_prompt_tokens == 300
        assert analysis.total_completion_tokens == 150
        assert "gpt-4o" in analysis.models_used

    def test_mixed_models(self, tmp_path):
        for model in ["gpt-4o", "gpt-4o-mini"]:
            jsonl = tmp_path / f"{model}_temp.jsonl"
            record = {
                "status": "success",
                "response_data": {
                    "model": model,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                },
            }
            jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        files = sorted(tmp_path.glob("*_temp.jsonl"))
        analysis = perform_cost_analysis(files)

        assert len(analysis.models_used) == 2


# ---------------------------------------------------------------------------
# save_analysis_to_csv
# ---------------------------------------------------------------------------

class TestSaveAnalysisToCsv:
    def test_basic_csv(self, tmp_path):
        analysis = CostAnalysis(
            file_stats=[
                FileStats(
                    file_path=Path("test.jsonl"),
                    model="gpt-4o",
                    total_chunks=5,
                    successful_chunks=4,
                    failed_chunks=1,
                    prompt_tokens=1000,
                    cached_tokens=200,
                    completion_tokens=500,
                    reasoning_tokens=0,
                    total_tokens=1500,
                    cost_standard=0.005,
                    cost_discounted=0.0025,
                )
            ],
            total_files=1,
            total_chunks=5,
            total_prompt_tokens=1000,
            total_cached_tokens=200,
            total_completion_tokens=500,
            total_reasoning_tokens=0,
            total_tokens=1500,
            total_cost_standard=0.005,
            total_cost_discounted=0.0025,
            models_used={"gpt-4o": 1},
        )
        csv_path = tmp_path / "analysis.csv"
        save_analysis_to_csv(analysis, csv_path)

        assert csv_path.exists()

        with csv_path.open("r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        # Header + 1 data row + blank row + summary row
        assert len(reader) >= 3
        assert reader[0][0] == "File"
        assert reader[1][0] == "test.jsonl"

    def test_empty_analysis(self, tmp_path):
        analysis = CostAnalysis()
        csv_path = tmp_path / "empty.csv"
        save_analysis_to_csv(analysis, csv_path)
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    def test_token_usage_defaults(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.model == ""

    def test_file_stats_defaults(self):
        stats = FileStats(file_path=Path("test.jsonl"))
        assert stats.total_chunks == 0
        assert stats.cost_standard == 0.0

    def test_cost_analysis_defaults(self):
        analysis = CostAnalysis()
        assert analysis.total_files == 0
        assert analysis.models_used == {}
