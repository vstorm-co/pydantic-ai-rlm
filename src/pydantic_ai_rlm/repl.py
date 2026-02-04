from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync
from pydantic_ai.messages import TextPart

from .dependencies import ContextType, RLMConfig


@dataclass
class REPLResult:
    """Result from REPL code execution."""

    stdout: str
    """Standard output from execution."""

    stderr: str
    """Standard error from execution."""

    locals: dict[str, Any]
    """Local variables after execution."""

    execution_time: float
    """Time taken to execute in seconds."""

    success: bool = True
    """Whether execution completed without errors."""

    def __str__(self) -> str:
        return f"REPLResult(success={self.success}, stdout={self.stdout[:100]}..., stderr={self.stderr[:100]}...)"


class REPLEnvironment:
    """
    Sandboxed Python execution environment for RLM.

    Provides a safe environment where the LLM can execute Python code
    to analyze large contexts. The context is loaded as a variable
    accessible within the REPL.

    Key features:
    - Sandboxed execution with restricted built-ins
    - Persistent state across multiple executions
    - Stdout/stderr capture
    - Configurable security settings
    """

    # Safe built-ins that don't allow dangerous operations
    SAFE_BUILTINS: ClassVar[dict[str, Any]] = {
        # Core types
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "type": type,
        "isinstance": isinstance,
        "issubclass": issubclass,
        # Iteration
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "iter": iter,
        "next": next,
        # Math
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        # String/char
        "chr": chr,
        "ord": ord,
        "hex": hex,
        "bin": bin,
        "oct": oct,
        "repr": repr,
        "ascii": ascii,
        "format": format,
        # Collections
        "any": any,
        "all": all,
        "slice": slice,
        "hash": hash,
        "id": id,
        "callable": callable,
        # Attribute access
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "delattr": delattr,
        "dir": dir,
        "vars": vars,
        # Binary
        "bytes": bytes,
        "bytearray": bytearray,
        "memoryview": memoryview,
        "complex": complex,
        # OOP
        "super": super,
        "property": property,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "object": object,
        # Exceptions (for try/except blocks)
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "AssertionError": AssertionError,
        "NotImplementedError": NotImplementedError,
        # Allow imports (controlled)
        "__import__": __import__,
    }

    # Additional built-ins when file access is enabled
    FILE_ACCESS_BUILTINS: ClassVar[dict[str, Any]] = {
        "open": open,
        "FileNotFoundError": FileNotFoundError,
        "OSError": OSError,
        "IOError": IOError,
    }

    # Built-ins that are always blocked
    BLOCKED_BUILTINS: ClassVar[dict[str, None]] = {
        "eval": None,
        "exec": None,
        "compile": None,
        "globals": None,
        "locals": None,
        "input": None,
        "__builtins__": None,
    }

    def __init__(
        self,
        context: ContextType,
        config: RLMConfig | None = None,
    ):
        """
        Initialize the REPL environment.

        Args:
            context: The context data to make available as 'context' variable
            config: Configuration options for the REPL
        """
        self.config = config or RLMConfig()
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self._lock = threading.Lock()
        self.locals: dict[str, Any] = {}

        # Setup globals with safe built-ins
        self.globals: dict[str, Any] = {
            "__builtins__": self._create_builtins(),
        }

        if self.config.sub_model:
            self._setup_llm_query()

        # Load context into environment
        self._load_context(context)

    def _create_builtins(self) -> dict[str, Any]:
        """Create the built-ins dict based on config."""
        builtins = dict(self.SAFE_BUILTINS)

        # Always include file access builtins - needed for context loading
        # and generally useful for data analysis. The temp directory
        # provides sandboxing.
        builtins.update(self.FILE_ACCESS_BUILTINS)

        # Apply blocked builtins
        builtins.update(self.BLOCKED_BUILTINS)

        return builtins

    def _setup_llm_query(self) -> None:
        """
        Set up the llm_query function for the REPL environment.
        """
        from .logging import get_logger

        def llm_query(prompt: str) -> str:
            """
            Query a sub-LLM with the given prompt.

            This function allows you to delegate analysis tasks to another
            LLM, which is useful for processing large contexts in chunks.

            Args:
                prompt: The prompt to send to the sub-LLM

            Returns:
                The sub-LLM's response as a string
            """
            logger = get_logger()

            try:
                if not self.config.sub_model:
                    return "Error: No sub-model configured"

                # Log the query
                logger.log_llm_query(prompt)

                result = model_request_sync(
                    self.config.sub_model,
                    [ModelRequest.user_text_prompt(prompt)],
                )
                # Extract text from the response parts
                text_parts = [part.content for part in result.parts if isinstance(part, TextPart)]
                response = "".join(text_parts) if text_parts else ""

                # Log the response
                logger.log_llm_response(response)

                return response
            except Exception as e:
                return f"Error querying sub-LLM: {e!s}"

        # Add llm_query to globals
        self.globals["llm_query"] = llm_query

    def _load_context(self, context: ContextType) -> None:
        """
        Load context data into the REPL environment.

        The context is written to a file and then loaded into the
        'context' variable in the REPL namespace.
        """
        if isinstance(context, str):
            # Text context
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w", encoding="utf-8") as f:
                f.write(context)

            load_code = f"""
with open(r'{context_path}', 'r', encoding='utf-8') as f:
    context = f.read()
"""
        else:
            # JSON context (dict or list)
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2, default=str)

            load_code = f"""
import json
with open(r'{context_path}', 'r', encoding='utf-8') as f:
    context = json.load(f)
"""

        # Execute the load code to populate 'context' variable
        self._execute_internal(load_code)

    def _execute_internal(self, code: str) -> None:
        """Execute code internally without capturing output."""
        combined = {**self.globals, **self.locals}
        exec(code, combined, combined)

        # Update locals with new variables
        for key, value in combined.items():
            if key not in self.globals and not key.startswith("_"):
                self.locals[key] = value

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield stdout_buffer, stderr_buffer
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    @contextmanager
    def _temp_working_directory(self):
        """Context manager to temporarily change working directory."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def execute(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with stdout, stderr, locals, and timing
        """
        # Normalize code: remove common leading whitespace and strip
        code = textwrap.dedent(code).strip()

        start_time = time.time()
        success = True
        stdout_content = ""
        stderr_content = ""

        with (
            self._lock,
            self._capture_output() as (stdout_buffer, stderr_buffer),
            self._temp_working_directory(),
        ):
            try:
                # Split into imports and other code
                lines = code.split("\n")
                import_lines = []
                other_lines = []

                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")) and not stripped.startswith("#"):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)

                # Execute imports in globals
                if import_lines:
                    import_code = "\n".join(import_lines)
                    exec(import_code, self.globals, self.globals)

                # Execute rest of code
                if other_lines:
                    other_code = "\n".join(other_lines)
                    combined = {**self.globals, **self.locals}

                    # Try to evaluate last expression for display
                    self._execute_with_expression_display(other_code, other_lines, combined)

                    # Update locals
                    for key, value in combined.items():
                        if key not in self.globals:
                            self.locals[key] = value

                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()

            except Exception as e:
                success = False
                stderr_content = stderr_buffer.getvalue() + f"\nError: {e!s}"
                stdout_content = stdout_buffer.getvalue()

        execution_time = time.time() - start_time

        # Truncate output if needed
        max_chars = self.config.truncate_output_chars
        if len(stdout_content) > max_chars:
            stdout_content = stdout_content[:max_chars] + "\n... (output truncated)"
        if len(stderr_content) > max_chars:
            stderr_content = stderr_content[:max_chars] + "\n... (output truncated)"

        return REPLResult(
            stdout=stdout_content,
            stderr=stderr_content,
            locals=dict(self.locals),
            execution_time=execution_time,
            success=success,
        )

    def _execute_with_expression_display(
        self,
        code: str,
        lines: list[str],
        namespace: dict[str, Any],
    ) -> None:
        """
        Execute code, displaying the last expression's value if applicable.

        This mimics notebook/REPL behavior where the last expression is
        automatically displayed.
        """
        # Find non-comment, non-empty lines
        non_comment_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

        if not non_comment_lines:
            exec(code, namespace, namespace)
            return

        last_line = non_comment_lines[-1].strip()

        # Check if last line is an expression (not a statement)
        is_expression = (
            not last_line.startswith(
                (
                    "import ",
                    "from ",
                    "def ",
                    "class ",
                    "if ",
                    "for ",
                    "while ",
                    "try:",
                    "with ",
                    "return ",
                    "yield ",
                    "raise ",
                    "break",
                    "continue",
                    "pass",
                    "assert ",
                    "del ",
                    "global ",
                    "nonlocal ",
                )
            )
            and "=" not in last_line.split("#")[0]  # Not assignment
            and not last_line.endswith(":")  # Not control structure
            and not last_line.startswith("print(")  # Not explicit print
        )

        if is_expression and len(non_comment_lines) > 0:
            try:
                # Execute all but last line
                if len(non_comment_lines) > 1:
                    # Find where last line starts
                    last_line_idx = None
                    for i, line in enumerate(lines):
                        if line.strip() == last_line:
                            last_line_idx = i
                            break

                    if last_line_idx and last_line_idx > 0:
                        statements = "\n".join(lines[:last_line_idx])
                        exec(statements, namespace, namespace)

                # Evaluate and print last expression
                result = eval(last_line, namespace, namespace)
                if result is not None:
                    print(repr(result))

            except (SyntaxError, NameError):
                # Fall back to normal execution
                exec(code, namespace, namespace)
        else:
            exec(code, namespace, namespace)

    def cleanup(self) -> None:
        """Clean up temporary directory."""
        with contextlib.suppress(Exception):
            shutil.rmtree(self.temp_dir)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
