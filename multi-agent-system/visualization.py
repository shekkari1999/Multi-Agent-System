"""Graph visualization helpers."""

from IPython.display import Image


def show_graph(graph, xray: bool = False) -> Image:
    """Render a LangGraph diagram with a fallback draw method."""
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception:
        import nest_asyncio
        from langchain_core.runnables.graph import MermaidDrawMethod

        nest_asyncio.apply()
        return Image(
            graph.get_graph(xray=xray).draw_mermaid_png(
                draw_method=MermaidDrawMethod.PYPPETEER
            )
        )
