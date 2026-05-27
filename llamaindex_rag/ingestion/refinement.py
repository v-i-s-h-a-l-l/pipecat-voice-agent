import asyncio
import json

from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode
from llama_index.core.llms import LLM

REFINE_PROMPT = """
Rewrite this restaurant information clearly for customers.

Also generate 2 customer questions this could answer.

Return valid JSON only:

{
  "enriched_text": "...",
  "hyde_questions": ["...", "..."]
}

TEXT:
{text}
"""

_CONCURRENCY = 5


class LLMRefinementExtractor(BaseExtractor):

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    async def aextract(
        self,
        nodes: list[BaseNode],
    ) -> list[dict]:

        semaphore = asyncio.Semaphore(_CONCURRENCY)

        async def refine_one(node):

            async with semaphore:

                try:

                    response = await self._llm.acomplete(
                        REFINE_PROMPT.format(
                            text=node.get_content()[:500]
                        )
                    )

                    data = json.loads(response.text)

                    return {
                        "enriched_text": data.get(
                            "enriched_text",
                            "",
                        ),
                    }

                except Exception:
                    return {
                        "enriched_text": "",
                    }

        results = await asyncio.gather(
            *[refine_one(n) for n in nodes]
        )

        for node, meta in zip(nodes, results):

            if meta["enriched_text"]:
                node.set_content(meta["enriched_text"])

        return results