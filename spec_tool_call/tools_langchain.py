"""Tool definitions using LangChain @tool decorator."""
from langchain.tools import tool

# Import existing implementations
from .tools.search_tool import search_serper_web, search_serper_with_content
from .tools.file_tool import read_file as read_file_enhanced
from .tools.code_exec_tool import execute_python_code, execute_calculation, generate_python_code
from .tools.vision_tool import analyze_image, extract_text_from_image


# TEMPORARILY COMMENTED OUT - Using only search_with_content to simplify speculation
# @tool
# def search_web(query: str, expand_search: bool = False) -> str:
#     """Search the web and return quick results (titles and snippets).
#     
#     Fast search for finding relevant pages. Returns 3 results by default,
#     or 10 results if you need more options.
#     
#     Args:
#         query: The search query
#         expand_search: If True, return 10 results instead of 3 (use when you need more options)
#     """
#     max_results = 10 if expand_search else 3
#     return search_serper_web(query, max_results)


@tool
def search_with_content(query: str, expand_search: bool = False) -> str:
    """Search the web and extract full content from pages.
    
    Slower but more thorough - fetches and extracts actual page content.
    Returns 1 result by default (most relevant), or 3 if you need alternatives.
    
    Args:
        query: The search query
        expand_search: If True, return 3 results instead of 1 (use when confused or need alternatives)
    """
    max_results = 3 if expand_search else 1
    return search_serper_with_content(query, max_results)


@tool
def file_read(path: str) -> str:
    """Read a file and return its content.
    
    Supports various file types: .txt, .md, .json, .csv, .xlsx, .pdf, .docx, etc.
    Returns the extracted text content of the file.
    
    Args:
        path: Path to the file (use absolute path if provided in the question)
    """
    result = read_file_enhanced(path)
    
    if isinstance(result, dict):
        # Check for error
        if "error" in result:
            return f"Error: {result.get('message', result.get('error'))}"
        
        # Extract content based on file type
        kind = result.get("kind", "unknown")
        
        if kind == "text":
            return result.get("full_text") or result.get("preview", "")
        
        elif kind == "docx":
            # For DOCX, return full text
            content = result.get("full_text") or result.get("text_preview", "")
            paragraphs = result.get("paragraphs", 0)
            char_count = result.get("char_count", len(content))
            return f"DOCX Content ({paragraphs} paragraphs, {char_count} characters):\n\n{content}"
        
        elif kind == "pdf":
            content = result.get("text_preview", "")
            pages = result.get("pages", 0)
            return f"PDF Content ({pages} pages):\n\n{content}"
        
        elif kind == "json":
            return result.get("preview", str(result.get("data", "")))
        
        elif kind == "csv" or kind == "excel":
            # For structured data, format nicely
            rows = result.get("preview_rows", [])
            columns = result.get("column_names", [])
            return f"Columns: {columns}\n\nData:\n{rows}"
        
        else:
            # Return string representation
            return str(result)
    
    return str(result)


@tool
def calculate(expression: str) -> str:
    """Evaluate a single mathematical expression.
    
    For SINGLE expressions only (no variables or multiple statements).
    Use Python syntax: 2+3, 10*5, 2**10, sqrt(16), (356400 / 42.195) / (2 + 1/60 + 9/3600)
    
    For multi-step calculations with variables, use code_exec instead.
    
    Args:
        expression: Single Python expression to evaluate
    """
    result = execute_calculation(expression)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return f"Result: {result.get('result')}"
        return f"Error: {result.get('error')}\nExpression was: {expression}"
    return str(result)


@tool
def code_generate(task_description: str, context: str = None) -> str:
    """Generate Python code using GPT-5.
    
    Args:
        task_description: What the code should do
        context: Optional additional context
    """
    result = generate_python_code(task_description, context)
    if isinstance(result, dict) and result.get("status") == "success":
        return result.get("code", "")
    return str(result)


@tool
def code_exec(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandbox.
    
    Args:
        code: Python code to execute
        timeout: Timeout in seconds
    """
    result = execute_python_code(code, timeout)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("output", "")
        return f"Error: {result.get('error')}"
    return str(result)


@tool
def vision_analyze(image_path: str, question: str = None) -> str:
    """Analyze an image using vision model.
    
    Args:
        image_path: Path to the image
        question: Optional question about the image
    """
    result = analyze_image(image_path, question)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("analysis", "")
        return f"Error: {result.get('error')}"
    return str(result)


@tool
def vision_ocr(image_path: str) -> str:
    """Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image
    """
    result = extract_text_from_image(image_path)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("text", "")
        return f"Error: {result.get('error')}"
    return str(result)


# All tools
ALL_TOOLS = [
    # search_web,  # Commented out - using only search_with_content
    search_with_content,
    file_read,
    calculate,
    code_generate,
    code_exec,
    vision_analyze,
    vision_ocr,
]

# Tools by name
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}

# Read-only tools (safe for speculation)
READ_ONLY_TOOLS = [
    # search_web,  # Commented out - using only search_with_content
    search_with_content,
    file_read,
    calculate,
    code_generate,
    vision_analyze,
    vision_ocr,
]
