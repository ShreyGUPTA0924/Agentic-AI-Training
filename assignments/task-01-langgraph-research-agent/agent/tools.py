from datetime import datetime
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import ast
import operator


# -------------------------
# Safe Calculator Helpers
# -------------------------

operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(node):

    if isinstance(node, ast.Constant):
        return node.value

    elif isinstance(node, ast.BinOp):

        left = safe_eval(node.left)
        right = safe_eval(node.right)

        return operators[type(node.op)](
            left,
            right
        )

    elif isinstance(node, ast.UnaryOp):

        operand = safe_eval(node.operand)

        return operators[type(node.op)](
            operand
        )

    raise TypeError("Unsupported expression")


# -------------------------
# Current Date Tool
# -------------------------

@tool
def get_current_date() -> str:
    """
    Returns today's date.
    """

    return datetime.now().strftime(
        "%Y-%m-%d"
    )


# -------------------------
# Web Search Tool
# -------------------------

@tool
def search_web(query: str) -> list:
    """
    Search the web and return top search results.
    """

    try:

        results = []

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=5
            )

            for result in search_results:

                results.append(
                    {
                        "title": result.get(
                            "title",
                            ""
                        ),
                        "snippet": result.get(
                            "body",
                            ""
                        ),
                        "url": result.get(
                            "href",
                            ""
                        )
                    }
                )

        return results

    except Exception as e:

        return [
            {
                "error": str(e)
            }
        ]


# -------------------------
# Calculator Tool
# -------------------------

@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    """

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = safe_eval(
            tree.body
        )

        return str(result)

    except Exception as e:

        return (
            f"Calculation error: {e}"
        )