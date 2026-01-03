import asyncio

from vkusvillbot.config import Settings
from vkusvillbot.db import Database
from vkusvillbot.embeddings_client import OpenRouterEmbeddingsClient
from vkusvillbot.manual_llm import ManualLLM
from vkusvillbot.mcp_client import VkusvillMCP
from vkusvillbot.models import UserProfile
from vkusvillbot.product_retriever import ProductRetriever
from vkusvillbot.sgr_agent import SgrAgent, SgrConfig
from vkusvillbot.vector_index import FaissVectorIndex


async def main() -> None:
    settings = Settings.load()
    db = Database(settings.db.path)
    if db.has_products():
        db.ensure_product_columns()
        db.ensure_fts()

    embeddings = OpenRouterEmbeddingsClient(
        api_key=settings.llm.api_key,
        model=settings.vector.embedding_model,
        referer=settings.llm.http_referer,
        title=settings.llm.title,
        proxy_url=settings.llm.proxy_url,
    )
    retriever = ProductRetriever(
        db=db,
        embeddings=embeddings,
        index=FaissVectorIndex(settings.vector.index_path),
        candidate_pool=settings.vector.candidate_pool,
        fts_boost=settings.vector.fts_boost,
    )

    mcp = VkusvillMCP(settings.mcp.url)
    await mcp.connect()
    try:
        llm = ManualLLM()
        agent = SgrAgent(
            mcp=mcp,
            llm=llm,
            db=db,
            retriever=retriever,
            config=SgrConfig(max_steps=6, max_items_per_search=8, temperature=0.3),
            profile=UserProfile(city="Москва", diet_notes=None),
        )
        text = input("USER > ")
        reply = await agent.run(text)
        print("\nANSWER:")
        print(reply)
    finally:
        await mcp.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
