# modules/ui/cost_display.py

"""
Display functions for cost analysis results.
"""

from typing import Optional

from modules.operations.cost_analysis import CostAnalysis
from modules.ui.core import UserInterface


def display_analysis(analysis: CostAnalysis, ui: Optional[UserInterface] = None) -> None:
    """
    Display cost analysis results.
    
    Args:
        analysis: CostAnalysis object
        ui: Optional UserInterface for formatted output
    """
    if ui:
        ui.print_section_header("Token Cost Analysis")
        ui.print_info(f"Total files analyzed: {analysis.total_files}")
        ui.print_info(f"Total chunks processed: {analysis.total_chunks}")
        ui.print_info("")
        
        ui.print_info("Token Usage:")
        ui.print_info(f"  Prompt tokens: {analysis.total_prompt_tokens:,}")
        ui.print_info(f"  Cached tokens: {analysis.total_cached_tokens:,}")
        ui.print_info(f"  Completion tokens: {analysis.total_completion_tokens:,}")
        if analysis.total_reasoning_tokens > 0:
            ui.print_info(f"  Reasoning tokens: {analysis.total_reasoning_tokens:,}")
        ui.print_info(f"  Total tokens: {analysis.total_tokens:,}")
        ui.print_info("")
        
        ui.print_info("Models Used:")
        for model, count in analysis.models_used.items():
            ui.print_info(f"  {model}: {count} file(s)")
        ui.print_info("")
        
        ui.print_section_header("Cost Estimates")
        ui.print_info(f"Standard pricing: ${analysis.total_cost_standard:.4f}")
        ui.print_success(f"Discounted pricing (50% off): ${analysis.total_cost_discounted:.4f}")
        ui.print_info(f"Potential savings: ${analysis.total_cost_standard - analysis.total_cost_discounted:.4f}")
        ui.print_info("")
        
        if analysis.file_stats:
            ui.print_section_header("Per-File Breakdown")
            for stats in analysis.file_stats:
                ui.print_info(f"\n{stats.file_path.name}:")
                ui.print_info(f"  Model: {stats.model or 'Unknown'}")
                ui.print_info(f"  Chunks: {stats.successful_chunks}/{stats.total_chunks} successful")
                ui.print_info(f"  Tokens: {stats.total_tokens:,}")
                ui.print_info(f"  Cost (standard): ${stats.cost_standard:.4f}")
                ui.print_info(f"  Cost (discounted): ${stats.cost_discounted:.4f}")
    else:
        # CLI mode output
        print("\n=== Token Cost Analysis ===")
        print(f"Total files analyzed: {analysis.total_files}")
        print(f"Total chunks processed: {analysis.total_chunks}")
        print("")
        
        print("Token Usage:")
        print(f"  Prompt tokens: {analysis.total_prompt_tokens:,}")
        print(f"  Cached tokens: {analysis.total_cached_tokens:,}")
        print(f"  Completion tokens: {analysis.total_completion_tokens:,}")
        if analysis.total_reasoning_tokens > 0:
            print(f"  Reasoning tokens: {analysis.total_reasoning_tokens:,}")
        print(f"  Total tokens: {analysis.total_tokens:,}")
        print("")
        
        print("Models Used:")
        for model, count in analysis.models_used.items():
            print(f"  {model}: {count} file(s)")
        print("")
        
        print("=== Cost Estimates ===")
        print(f"Standard pricing: ${analysis.total_cost_standard:.4f}")
        print(f"Discounted pricing (50% off): ${analysis.total_cost_discounted:.4f}")
        print(f"Potential savings: ${analysis.total_cost_standard - analysis.total_cost_discounted:.4f}")
        print("")
        
        if analysis.file_stats:
            print("=== Per-File Breakdown ===")
            for stats in analysis.file_stats:
                print(f"\n{stats.file_path.name}:")
                print(f"  Model: {stats.model or 'Unknown'}")
                print(f"  Chunks: {stats.successful_chunks}/{stats.total_chunks} successful")
                print(f"  Tokens: {stats.total_tokens:,}")
                print(f"  Cost (standard): ${stats.cost_standard:.4f}")
                print(f"  Cost (discounted): ${stats.cost_discounted:.4f}")
