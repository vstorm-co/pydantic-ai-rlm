from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .repl import REPLResult

# Check if rich is available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RLMLogger:
    """
    Pretty logger for RLM code execution.

    Uses rich for fancy terminal output with syntax highlighting and styled panels.
    Falls back to plain text if rich is not installed.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def log_code_execution(self, code: str) -> None:
        """Log the code being executed."""
        if not self.enabled:
            return

        if RICH_AVAILABLE and self.console:
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            panel = Panel(
                syntax,
                title="[bold cyan]Code Execution[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
            self.console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("CODE EXECUTION")
            print("=" * 50)
            print(code)
            print("=" * 50)

    def log_result(self, result: REPLResult) -> None:
        """Log the execution result."""
        if not self.enabled:
            return

        if RICH_AVAILABLE and self.console:
            self._log_result_rich(result)
        else:
            self._log_result_plain(result)

    def _log_result_rich(self, result: REPLResult) -> None:
        """Log result using rich formatting."""
        status, border_style = self._get_status_style(result.success)
        content_parts = self._build_content_parts(result)
        user_vars = self._get_user_vars(result.locals)

        self._print_result_panel(content_parts, status, border_style, user_vars)

    def _get_status_style(self, success: bool) -> tuple:
        """Get status text and border style based on success."""
        if success:
            return Text("SUCCESS", style="bold green"), "green"
        return Text("ERROR", style="bold red"), "red"

    def _build_content_parts(self, result: REPLResult) -> list:
        """Build content parts for the result panel."""
        parts = [Text(f"Executed in {result.execution_time:.3f}s", style="dim")]

        if result.stdout.strip():
            stdout = result.stdout.strip()
            if len(stdout) > 2000:
                stdout = stdout[:2000] + "\n... (truncated)"
            parts.extend([Text("\n"), Text("Output:", style="bold yellow"), Text("\n"), Text(stdout, style="white")])

        if result.stderr.strip():
            stderr = result.stderr.strip()
            if len(stderr) > 1000:
                stderr = stderr[:1000] + "\n... (truncated)"
            parts.extend([Text("\n"), Text("Errors:", style="bold red"), Text("\n"), Text(stderr, style="red")])

        return parts

    def _get_user_vars(self, locals_dict: dict) -> dict:
        """Extract user-defined variables from locals."""
        excluded = ("context", "json", "re", "os", "collections", "math")
        return {k: v for k, v in locals_dict.items() if not k.startswith("_") and k not in excluded}

    def _print_result_panel(self, content_parts: list, status, border_style: str, user_vars: dict) -> None:
        """Print the result panel and optional variables table."""
        from rich.table import Table

        if user_vars:
            content_parts.extend([Text("\n"), Text("Variables:", style="bold magenta"), Text("\n")])
            if len(user_vars) > 10:
                content_parts.append(Text(f"  ... and {len(user_vars) - 10} more variables\n", style="dim"))

        combined = Text()
        for part in content_parts:
            combined.append(part)

        panel = Panel(combined, title=f"[bold]Result: {status}[/bold]", border_style=border_style, padding=(0, 1))
        self.console.print(panel)

        if user_vars:
            var_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
            var_table.add_column("Name", style="cyan")
            var_table.add_column("Type", style="yellow")
            var_table.add_column("Value", style="white", max_width=60)

            for name, value in list(user_vars.items())[:10]:
                value_str = self._format_var_value(value)
                var_table.add_row(name, type(value).__name__, value_str)

            self.console.print(var_table)

    def _format_var_value(self, value) -> str:
        """Format a variable value for display."""
        try:
            value_str = repr(value)
            if len(value_str) > 60:
                return value_str[:57] + "..."
            return value_str
        except Exception:
            return "<unable to repr>"

    def _log_result_plain(self, result: REPLResult) -> None:
        """Log result using plain text."""
        status = "SUCCESS" if result.success else "ERROR"
        print(f"\n{'=' * 50}")
        print(f"RESULT: {status} (executed in {result.execution_time:.3f}s)")
        print("=" * 50)

        if result.stdout.strip():
            print("\nOutput:")
            stdout = result.stdout.strip()
            if len(stdout) > 2000:
                stdout = stdout[:2000] + "\n... (truncated)"
            print(stdout)

        if result.stderr.strip():
            print("\nErrors:")
            stderr = result.stderr.strip()
            if len(stderr) > 1000:
                stderr = stderr[:1000] + "\n... (truncated)"
            print(stderr)

        user_vars = {k: v for k, v in result.locals.items() if not k.startswith("_") and k not in ("context", "json", "re", "os")}
        if user_vars:
            print("\nVariables:")
            for name, value in list(user_vars.items())[:10]:
                try:
                    value_str = repr(value)
                    if len(value_str) > 60:
                        value_str = value_str[:57] + "..."
                except Exception:
                    value_str = "<unable to repr>"
                print(f"  {name} ({type(value).__name__}): {value_str}")
            if len(user_vars) > 10:
                print(f"  ... and {len(user_vars) - 10} more variables")

        print("=" * 50)

    def log_llm_query(self, prompt: str) -> None:
        """Log an llm_query call."""
        if not self.enabled:
            return

        if RICH_AVAILABLE and self.console:
            # Truncate long prompts
            display_prompt = prompt
            if len(display_prompt) > 500:
                display_prompt = display_prompt[:500] + "..."

            panel = Panel(
                Text(display_prompt, style="white"),
                title="[bold blue]LLM Query[/bold blue]",
                border_style="blue",
                padding=(0, 1),
            )
            self.console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("LLM QUERY")
            print("=" * 50)
            display_prompt = prompt
            if len(display_prompt) > 500:
                display_prompt = display_prompt[:500] + "..."
            print(display_prompt)
            print("=" * 50)

    def log_llm_response(self, response: str) -> None:
        """Log an llm_query response."""
        if not self.enabled:
            return

        if RICH_AVAILABLE and self.console:
            # Truncate long responses
            display_response = response
            if len(display_response) > 500:
                display_response = display_response[:500] + "..."

            panel = Panel(
                Text(display_response, style="white"),
                title="[bold blue]LLM Response[/bold blue]",
                border_style="blue",
                padding=(0, 1),
            )
            self.console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("LLM RESPONSE")
            print("=" * 50)
            display_response = response
            if len(display_response) > 500:
                display_response = display_response[:500] + "..."
            print(display_response)
            print("=" * 50)


# Global logger instance
_logger: RLMLogger | None = None


def get_logger() -> RLMLogger:
    """Get the global RLM logger instance."""
    global _logger
    if _logger is None:
        _logger = RLMLogger(enabled=False)  # Disabled by default
    return _logger


def configure_logging(enabled: bool = True) -> RLMLogger:
    """
    Configure RLM logging.

    Args:
        enabled: Whether to enable logging output

    Returns:
        The configured logger instance

    Example:
        ```python
        from pydantic_ai_rlm import configure_logging

        # Enable fancy logging
        configure_logging(enabled=True)

        # Run your analysis - you'll see code and output in the terminal
        result = await run_rlm_analysis(context, query)
        ```
    """
    global _logger
    _logger = RLMLogger(enabled=enabled)
    return _logger
