"""Tests for context image resolution and injection."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.config.context import resolve_context_image_for_extraction

# ---------------------------------------------------------------------------
# Resolution hierarchy tests
# ---------------------------------------------------------------------------


class TestResolveContextImageForExtraction:
    """Test hierarchical context image resolution."""

    @pytest.mark.unit
    def test_file_specific_png_found(self, tmp_path: Path) -> None:
        """File-specific context image (.png) is resolved at level 1."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        ctx_img = tmp_path / "corpus" / "page001_extract_context.png"
        ctx_img.write_bytes(b"fake context image")

        result_path, resolved = resolve_context_image_for_extraction(
            text_file=input_file
        )
        assert result_path == ctx_img
        assert resolved == ctx_img

    @pytest.mark.unit
    def test_file_specific_jpg_found(self, tmp_path: Path) -> None:
        """File-specific context image (.jpg) is resolved at level 1."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        ctx_img = tmp_path / "corpus" / "page001_extract_context.jpg"
        ctx_img.write_bytes(b"fake context image")

        result_path, resolved = resolve_context_image_for_extraction(
            text_file=input_file
        )
        assert result_path == ctx_img
        assert resolved == ctx_img

    @pytest.mark.unit
    def test_png_takes_priority_over_jpg(self, tmp_path: Path) -> None:
        """When both .png and .jpg exist, .png wins (earlier in priority)."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        ctx_png = tmp_path / "corpus" / "page001_extract_context.png"
        ctx_png.write_bytes(b"png context")
        ctx_jpg = tmp_path / "corpus" / "page001_extract_context.jpg"
        ctx_jpg.write_bytes(b"jpg context")

        result_path, _ = resolve_context_image_for_extraction(text_file=input_file)
        assert result_path == ctx_png

    @pytest.mark.unit
    def test_folder_specific_fallback(self, tmp_path: Path) -> None:
        """Folder-specific context image is resolved at level 2."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        # No file-specific context image exists
        # Place folder-specific one level up
        ctx_img = tmp_path / "corpus_extract_context.png"
        ctx_img.write_bytes(b"folder context")

        result_path, resolved = resolve_context_image_for_extraction(
            text_file=input_file
        )
        assert result_path == ctx_img
        assert resolved == ctx_img

    @pytest.mark.unit
    def test_general_fallback(self, tmp_path: Path) -> None:
        """General fallback in context_dir is resolved at level 3."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        ctx_img = context_dir / "extract_context.jpg"
        ctx_img.write_bytes(b"general context")

        result_path, resolved = resolve_context_image_for_extraction(
            text_file=input_file, context_dir=context_dir
        )
        assert result_path == ctx_img
        assert resolved == ctx_img

    @pytest.mark.unit
    def test_no_context_image_returns_none(self, tmp_path: Path) -> None:
        """Returns (None, None) when no context image exists."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        context_dir = tmp_path / "context"
        context_dir.mkdir()

        result_path, resolved = resolve_context_image_for_extraction(
            text_file=input_file, context_dir=context_dir
        )
        assert result_path is None
        assert resolved is None

    @pytest.mark.unit
    def test_no_text_file_uses_general_only(self, tmp_path: Path) -> None:
        """When text_file is None, only general fallback is tried."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        ctx_img = context_dir / "extract_context.png"
        ctx_img.write_bytes(b"general context")

        result_path, _ = resolve_context_image_for_extraction(
            text_file=None, context_dir=context_dir
        )
        assert result_path == ctx_img

    @pytest.mark.unit
    def test_file_specific_wins_over_folder(self, tmp_path: Path) -> None:
        """File-specific takes priority over folder-specific."""
        input_file = tmp_path / "corpus" / "page001.png"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"fake image")

        # Both exist
        file_ctx = tmp_path / "corpus" / "page001_extract_context.png"
        file_ctx.write_bytes(b"file-specific")
        folder_ctx = tmp_path / "corpus_extract_context.png"
        folder_ctx.write_bytes(b"folder-specific")

        result_path, _ = resolve_context_image_for_extraction(text_file=input_file)
        assert result_path == file_ctx


# ---------------------------------------------------------------------------
# Preprocessing helper tests
# ---------------------------------------------------------------------------


class TestPreprocessContextImage:
    """Test _preprocess_context_image helper."""

    @pytest.mark.unit
    def test_produces_valid_dict(self, tmp_path: Path) -> None:
        """Returns dict with required keys and JPEG mime type."""
        from PIL import Image

        # Create a minimal test image
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test_context.png"
        img.save(img_path)

        from modules.extract.file_processor import _preprocess_context_image

        result = _preprocess_context_image(
            image_path=img_path,
            provider="openai",
            model_name="gpt-4o",
            image_detail="high",
        )

        assert "base64" in result
        assert "mime_type" in result
        assert "detail" in result
        assert result["mime_type"] == "image/jpeg"
        assert result["detail"] == "high"
        # Verify base64 is valid
        decoded = base64.b64decode(result["base64"])
        assert len(decoded) > 0

    @pytest.mark.unit
    def test_temp_directory_cleaned_up(self, tmp_path: Path) -> None:
        """Temp directory created during preprocessing is removed."""
        import tempfile

        from PIL import Image

        img = Image.new("RGB", (50, 50), color="blue")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Count temp dirs before
        temp_root = Path(tempfile.gettempdir())
        before = set(
            p for p in temp_root.iterdir() if p.name.startswith("chronominer_ctx_img_")
        )

        from modules.extract.file_processor import _preprocess_context_image

        _preprocess_context_image(
            image_path=img_path,
            provider="openai",
            model_name="gpt-4o",
            image_detail="auto",
        )

        after = set(
            p for p in temp_root.iterdir() if p.name.startswith("chronominer_ctx_img_")
        )
        # No new temp dirs should remain
        new_dirs = after - before
        assert len(new_dirs) == 0


# ---------------------------------------------------------------------------
# Integration: context image injection into messages
# ---------------------------------------------------------------------------


class TestContextImageInjection:
    """Test that context image is correctly injected into LLM messages."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_text_chunk_with_context_image(self) -> None:
        """Context image block appears in user message for OpenAI text."""
        from modules.llm.openai_utils import process_text_chunk

        mock_llm = AsyncMock()
        mock_llm.ainvoke_with_structured_output = AsyncMock(
            return_value={
                "output_text": "{}",
                "response_data": {},
                "request_metadata": {},
            }
        )

        mock_extractor = MagicMock()
        mock_extractor.provider = "openai"
        mock_extractor.caps.supports_structured_outputs = False
        mock_extractor.caps.supports_image_detail = True
        mock_extractor.llm = mock_llm

        ctx_data = {
            "base64": base64.b64encode(b"fake").decode(),
            "mime_type": "image/jpeg",
            "detail": "high",
        }

        await process_text_chunk(
            text_chunk="Some text content",
            extractor=mock_extractor,
            system_message="System prompt",
            context_image_data=ctx_data,
        )

        call_args = mock_llm.ainvoke_with_structured_output.call_args
        messages = call_args.kwargs["messages"]

        # User message should have 3 items: label, image block, text
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 3
        assert user_msg["content"][0] == {
            "type": "text",
            "text": "Context image:",
        }
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][2] == {
            "type": "input_text",
            "text": "Some text content",
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_image_chunk_with_context_image(self) -> None:
        """Context image block appears before main image for OpenAI."""
        from modules.llm.openai_utils import process_image_chunk

        mock_llm = AsyncMock()
        mock_llm.ainvoke_with_structured_output = AsyncMock(
            return_value={
                "output_text": "{}",
                "response_data": {},
                "request_metadata": {},
            }
        )

        mock_extractor = MagicMock()
        mock_extractor.provider = "openai"
        mock_extractor.caps.supports_structured_outputs = False
        mock_extractor.caps.supports_image_detail = True
        mock_extractor.llm = mock_llm

        ctx_data = {
            "base64": base64.b64encode(b"ctx_fake").decode(),
            "mime_type": "image/jpeg",
            "detail": "high",
        }

        await process_image_chunk(
            image_base64=base64.b64encode(b"main_fake").decode(),
            mime_type="image/jpeg",
            extractor=mock_extractor,
            system_message="System prompt",
            image_detail="high",
            context_image_data=ctx_data,
        )

        call_args = mock_llm.ainvoke_with_structured_output.call_args
        messages = call_args.kwargs["messages"]

        # User message: label, ctx image, instruction, main image
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 4
        assert user_msg["content"][0] == {
            "type": "text",
            "text": "Context image:",
        }
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][2]["type"] == "text"
        assert user_msg["content"][3]["type"] == "image_url"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_anthropic_skips_context_image(self) -> None:
        """Non-OpenAI provider logs warning and skips context image."""
        from modules.llm.openai_utils import process_text_chunk

        mock_llm = AsyncMock()
        mock_llm.ainvoke_with_structured_output = AsyncMock(
            return_value={
                "output_text": "{}",
                "response_data": {},
                "request_metadata": {},
            }
        )

        mock_extractor = MagicMock()
        mock_extractor.provider = "anthropic"
        mock_extractor.caps.supports_structured_outputs = False
        mock_extractor.caps.supports_image_detail = False
        mock_extractor.llm = mock_llm

        ctx_data = {
            "base64": base64.b64encode(b"fake").decode(),
            "mime_type": "image/jpeg",
            "detail": "high",
        }

        with patch("modules.llm.openai_utils.logger") as mock_logger:
            await process_text_chunk(
                text_chunk="Some text",
                extractor=mock_extractor,
                system_message="System",
                context_image_data=ctx_data,
            )
            mock_logger.warning.assert_called_once()
            assert "not yet supported" in mock_logger.warning.call_args[0][0]

        # User message should only have the text (no image block)
        call_args = mock_llm.ainvoke_with_structured_output.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]
        assert len(user_msg["content"]) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_context_image_preserves_existing_behavior(self) -> None:
        """Without context_image_data, messages are unchanged."""
        from modules.llm.openai_utils import process_text_chunk

        mock_llm = AsyncMock()
        mock_llm.ainvoke_with_structured_output = AsyncMock(
            return_value={
                "output_text": "{}",
                "response_data": {},
                "request_metadata": {},
            }
        )

        mock_extractor = MagicMock()
        mock_extractor.provider = "openai"
        mock_extractor.caps.supports_structured_outputs = False
        mock_extractor.llm = mock_llm

        await process_text_chunk(
            text_chunk="Some text",
            extractor=mock_extractor,
            system_message="System",
        )

        call_args = mock_llm.ainvoke_with_structured_output.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0] == {
            "type": "input_text",
            "text": "Some text",
        }


# ---------------------------------------------------------------------------
# CLI flag parsing tests
# ---------------------------------------------------------------------------


class TestCLIContextImageFlag:
    """Test --context-image CLI flag parsing."""

    @pytest.mark.unit
    def test_context_image_flag_true(self) -> None:
        """--context-image sets context_image to True."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args(
            ["--schema", "test", "--input", ".", "--context-image"]
        )
        assert args.context_image is True

    @pytest.mark.unit
    def test_context_image_flag_absent(self) -> None:
        """Absent --context-image defaults to False."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args(["--schema", "test", "--input", "."])
        assert args.context_image is False
