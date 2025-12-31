import asyncio

from vkusvillbot.manual_llm import ManualLLM
from vkusvillbot.mcp_client import VkusvillMCP
from vkusvillbot.models import UserProfile
from vkusvillbot.sgr_agent import SgrAgent, SgrConfig


async def main() -> None:
    mcp = VkusvillMCP("https://mcp001.vkusvill.ru/mcp")
    await mcp.connect()
    try:
        llm = ManualLLM()
        agent = SgrAgent(
            mcp=mcp,
            llm=llm,
            config=SgrConfig(max_steps=6, max_items_per_search=8, temperature=0.3),
            profile=UserProfile(city="Москва", diet_notes=None),
        )
        text = input("USER > ")
        reply = await agent.run(text)
        print("\nANSWER:")
        print(reply)
    finally:
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
