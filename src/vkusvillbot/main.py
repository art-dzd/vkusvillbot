from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from vkusvillbot.config import Settings
from vkusvillbot.db import Database
from vkusvillbot.formatting import to_telegram_markdown
from vkusvillbot.llm_client import OpenRouterClient
from vkusvillbot.logging import setup_logging
from vkusvillbot.mcp_client import MCPError, VkusvillMCP
from vkusvillbot.models import UserProfile
from vkusvillbot.sgr_agent import SgrAgent, SgrConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = Settings.load()
    setup_logging(settings.app.log_level)

    if not settings.telegram.token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

    db = Database(settings.db.path)
    mcp = VkusvillMCP(settings.mcp.url)
    await mcp.connect()

    llm = OpenRouterClient(
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        referer=settings.llm.http_referer,
        title=settings.llm.title,
        provider_order=settings.llm.provider_order,
        proxy_url=settings.llm.proxy_url,
    )

    sgr_config = SgrConfig(
        max_steps=settings.sgr.max_steps,
        max_items_per_search=settings.sgr.max_items_per_search,
        temperature=settings.sgr.temperature,
    )

    bot = Bot(token=settings.telegram.token)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        user = db.get_or_create_user(message.from_user.id)
        await message.answer(
            f"Привет! Я бот ВкусВилл. Город: {user.city}. Напишите запрос, например: 'молоко'."
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(
            "Команды:\n"
            "/diet — задать особенности питания\n"
            "/city — задать город\n"
            "Примеры: 'найди молоко', 'состав творога', 'собери корзину: хлеб молоко'"
        )

    @dp.message(Command("diet"))
    async def cmd_diet(message: Message) -> None:
        text = message.text.replace("/diet", "", 1).strip()
        if not text:
            await message.answer("Напишите особенности питания после команды /diet")
            return
        db.update_user_diet_notes(message.from_user.id, text)
        await message.answer("Сохранил особенности питания.")

    @dp.message(Command("city"))
    async def cmd_city(message: Message) -> None:
        text = message.text.replace("/city", "", 1).strip()
        if not text:
            await message.answer("Напишите город после команды /city")
            return
        db.update_user_city(message.from_user.id, text)
        await message.answer(f"Город обновлён: {text}")

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        user = db.get_or_create_user(message.from_user.id)
        profile = UserProfile(city=user.city, diet_notes=user.diet_notes)
        agent = SgrAgent(mcp=mcp, llm=llm, config=sgr_config, profile=profile)
        text = message.text or ""
        try:
            reply = await agent.run(text)
            db.save_session(user.id, "sgr", {"query": text})
            reply_md = to_telegram_markdown(reply)
            await message.answer(
                reply_md,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True,
            )
        except MCPError as exc:
            logger.exception("MCP error")
            await message.answer(f"Ошибка MCP: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error")
            await message.answer(f"Ошибка: {exc}")

    try:
        await dp.start_polling(bot)
    finally:
        await mcp.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
