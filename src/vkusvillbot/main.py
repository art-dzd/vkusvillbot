from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import cast

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

from vkusvillbot.config import Settings
from vkusvillbot.db import Database
from vkusvillbot.embeddings_client import OpenRouterEmbeddingsClient
from vkusvillbot.formatting import to_telegram_markdown
from vkusvillbot.llm_client import OpenRouterClient
from vkusvillbot.logging import setup_dialog_logger, setup_logging
from vkusvillbot.mcp_client import MCPError, VkusvillMCP
from vkusvillbot.models import UserProfile
from vkusvillbot.product_retriever import ProductRetriever
from vkusvillbot.sgr_agent import SgrAgent, SgrConfig
from vkusvillbot.telegram_draft import DraftProgress, TelegramAPI, TelegramAPIError
from vkusvillbot.vector_index import FaissVectorIndex

logger = logging.getLogger(__name__)


_TG_MAX_LEN = 4096


def _split_text(text: str, limit: int = _TG_MAX_LEN) -> list[str]:
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > limit:
            chunks.append(current)
            current = line
            continue
        current += line
    if current:
        chunks.append(current)
    return chunks


async def _typing_loop(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with suppress(Exception):
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except TimeoutError:
            continue


async def _pseudo_stream_plain(message: Message, text: str) -> None:
    if not text:
        return

    # Чтобы не попасть в лимит длины + не спамить editMessageText.
    preview = text.strip()
    if len(preview) > 3500:
        preview = preview[:3500].rstrip() + "\n…"

    # 6–8 апдейтов дают ощущение «пишет», но не добавляют большую задержку.
    max_updates = 8
    step = max(200, len(preview) // max_updates)
    last = 0
    for i in range(step, len(preview) + step, step):
        chunk = preview[:i]
        if len(chunk) == last:
            continue
        last = len(chunk)
        try:
            await message.edit_text(chunk, parse_mode=None, disable_web_page_preview=True)
        except TelegramBadRequest:
            break
        await asyncio.sleep(0.7)


async def main() -> None:
    settings = Settings.load()
    setup_logging(settings.app.log_level)
    dialog_logger = setup_dialog_logger()

    if not settings.telegram.token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

    tg_api = TelegramAPI(settings.telegram.token)
    topics_enabled = False
    try:
        me = await tg_api.get_me()
        topics_enabled = bool(me.get("has_topics_enabled"))
    except TelegramAPIError as exc:
        logger.warning("Не удалось вызвать getMe для проверки topics: %s", exc)

    db = Database(settings.db.path)
    if db.has_products():
        db.ensure_product_columns()
        db.ensure_fts()
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

    embeddings = OpenRouterEmbeddingsClient(
        api_key=settings.llm.api_key,
        model=settings.vector.embedding_model,
        referer=settings.llm.http_referer,
        title=settings.llm.title,
        proxy_url=settings.llm.proxy_url,
    )
    index = FaissVectorIndex(settings.vector.index_path)
    retriever = ProductRetriever(
        db=db,
        embeddings=embeddings,
        index=index,
        candidate_pool=settings.vector.candidate_pool,
        fts_boost=settings.vector.fts_boost,
    )

    sgr_config = SgrConfig(
        max_steps=settings.sgr.max_steps,
        max_items_per_search=settings.sgr.max_items_per_search,
        temperature=settings.sgr.temperature,
        history_messages=settings.sgr.history_messages,
        local_fresh_hours=settings.sgr.local_fresh_hours,
        use_mcp_refresh=settings.sgr.use_mcp_refresh,
    )

    bot = Bot(token=settings.telegram.token)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        user = db.get_or_create_user(message.from_user.id)
        topics_status = "включены" if topics_enabled else "выключены"
        streaming_status = (
            "включено" if (topics_enabled and settings.telegram.enable_drafts) else "выключено"
        )
        await message.answer(
            (
                f"Привет! Я бот ВкусВилл. Город: {user.city}.\n"
                f"Темы (forum mode) в личке: {topics_status}.\n"
                f"Стриминг через sendMessageDraft: {streaming_status}.\n\n"
                "Напишите запрос, например: 'молоко'."
            )
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
        use_drafts = bool(settings.telegram.enable_drafts and topics_enabled)
        draft: DraftProgress | None = None
        if use_drafts:
            draft = DraftProgress(
                api=tg_api,
                chat_id=cast(int, message.chat.id),
                draft_id=cast(int, message.message_id),
                message_thread_id=message.message_thread_id,
                enabled=bool(settings.telegram.show_progress),
            )
            try:
                await draft.set("🧠 Думаю…\n(прогресс появится ниже)")
            except TelegramAPIError as exc:
                logger.warning("sendMessageDraft недоступен, откатываюсь на fallback: %s", exc)
                draft = None
                use_drafts = False

        placeholder: Message | None = None
        stop_typing = asyncio.Event()
        typing_task: asyncio.Task[None] | None = None
        if not use_drafts:
            placeholder = await message.answer("Думаю…", disable_web_page_preview=True)
            typing_task = asyncio.create_task(_typing_loop(bot, message.chat.id, stop_typing))

        user = db.get_or_create_user(message.from_user.id)
        profile = UserProfile(city=user.city, diet_notes=user.diet_notes)
        agent = SgrAgent(
            mcp=mcp,
            llm=llm,
            db=db,
            retriever=retriever,
            config=sgr_config,
            profile=profile,
        )
        text = message.text or ""
        try:
            dialog_logger.info("USER tg_id=%s user_id=%s: %s", message.from_user.id, user.id, text)
            history = db.get_recent_messages(user.id, limit=settings.sgr.history_messages)
            progress_cb = draft.add if draft else None
            reply = await agent.run(text, history=history, user_id=user.id, progress=progress_cb)
            db.save_message(user.id, "user", text)
            db.save_message(user.id, "assistant", reply)
            dialog_logger.info("ASSISTANT user_id=%s: %s", user.id, reply)
            db.save_session(user.id, "sgr", {"query": text})

            if draft:
                with suppress(Exception):
                    await draft.add("✅ Готово, отправляю ответ…")

            parts = _split_text(reply)
            if len(parts) == 1:
                reply_md = to_telegram_markdown(reply)
                await message.answer(
                    reply_md,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    disable_web_page_preview=True,
                )
            else:
                await message.answer(
                    "Ответ слишком длинный — отправляю частями.",
                    disable_web_page_preview=True,
                )
                for part in parts:
                    part_md = to_telegram_markdown(part)
                    await message.answer(
                        part_md,
                        parse_mode=ParseMode.MARKDOWN_V2,
                        disable_web_page_preview=True,
                    )
        except MCPError as exc:
            logger.exception("MCP error")
            if placeholder:
                await placeholder.edit_text(f"Ошибка MCP: {exc}")
            else:
                await message.answer(f"Ошибка MCP: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error")
            if placeholder:
                await placeholder.edit_text(f"Ошибка: {exc}")
            else:
                await message.answer(f"Ошибка: {exc}")
        finally:
            stop_typing.set()
            if typing_task:
                typing_task.cancel()
                with suppress(asyncio.CancelledError):
                    await typing_task

    try:
        await dp.start_polling(bot)
    finally:
        await mcp.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
