import os
import sys
from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException
from crewai_tools import BaseTool

from pydantic import BaseModel



class MarkdownValidationToolSchema(BaseModel):
    file_path: str


class MarkdownValidationTool(BaseTool):
    name: str = "markdown_validation_tool"
    description: str = (
        "Validates markdown files and provides actionable feedback on syntax errors."
    )
    args_schema: type = MarkdownValidationToolSchema  # Define expected input schema

    def _run(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return "Could not validate file. The provided file path does not exist."

            scan_result = PyMarkdownApi().scan_path(file_path)
            return str(scan_result)
        except PyMarkdownApiException as e:
            return f"API Exception: {str(e)}"


# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python markdown_validation_tool.py <path_to_markdown_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    markdown_tool = MarkdownValidationTool()
    validation_results = markdown_tool.run(file_path)
    print("\nFinal Results:\n")
    print(validation_results)
