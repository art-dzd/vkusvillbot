from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import cast

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
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
_PENDING_TOPIC_TTL_S = 10.0


def _topic_ctx(
    message: Message,
    *,
    pending_routing: dict[str, int] | None = None,
) -> tuple[int, dict[str, int]]:
    """
    Возвращает:
    - key: идентификатор контекста (topic/thread) для истории
    - kwargs: параметры для отправки ответа в тот же контекст

    Telegram имеет 2 похожих механики:
    - message_thread_id: forum topics (в т.ч. topics в личке бота)
    - direct_messages_topic_id: direct messages chat topics (отдельная механика)

    Если Telegram не прислал идентификатор топика (часто бывает в первом сообщении),
    используем "reply thread": отвечаем как reply к корневому сообщению, чтобы ответы
    оставались внутри ветки "ответов" в клиенте.
    """

    if message.message_thread_id is not None:
        tid = int(message.message_thread_id)
        return tid, {"message_thread_id": tid}

    dm_topic = getattr(message, "direct_messages_topic", None)
    dm_tid = getattr(dm_topic, "topic_id", None)
    if dm_tid is not None:
        dm_tid_int = int(dm_tid)
        return dm_tid_int, {"direct_messages_topic_id": dm_tid_int}

    if pending_routing:
        pending_routing = dict(pending_routing)
        if "message_thread_id" in pending_routing:
            tid = int(pending_routing["message_thread_id"])
            return tid, {"message_thread_id": tid}
        if "direct_messages_topic_id" in pending_routing:
            tid = int(pending_routing["direct_messages_topic_id"])
            return tid, {"direct_messages_topic_id": tid}

    root_id = int(message.message_id)
    node = message.reply_to_message
    depth = 0
    while node is not None and depth < 8:
        root_id = int(node.message_id)
        node = getattr(node, "reply_to_message", None)
        depth += 1

    return root_id, {"reply_to_message_id": root_id}


def _split_text(text: str, limit: int = _TG_MAX_LEN) -> list[str]:
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break

        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit

        chunk = remaining[:cut]
        # Не заканчиваем chunk на одиночный backslash (важно для MarkdownV2 escapes).
        while chunk.endswith("\\") and len(chunk) > 1:
            cut -= 1
            chunk = remaining[:cut]

        chunk = chunk.strip("\n")
        if chunk:
            parts.append(chunk)
        remaining = remaining[cut:]
        remaining = remaining.lstrip("\n")

    return parts


def _reply_root_message_id(message: Message) -> int:
    root_id = int(message.message_id)
    node = message.reply_to_message
    depth = 0
    while node is not None and depth < 8:
        root_id = int(node.message_id)
        node = getattr(node, "reply_to_message", None)
        depth += 1
    return root_id


async def _safe_send(
    bot: Bot,
    *,
    chat_id: int,
    text: str,
    routing: dict[str, int],
    fallback_routing: dict[str, int],
    parse_mode: ParseMode | None = None,
    disable_web_page_preview: bool = True,
) -> tuple[Message, dict[str, int]]:
    try:
        msg = await bot.send_message(
            chat_id,
            text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            **routing,
        )
        return msg, routing
    except TelegramBadRequest as exc:
        logger.warning("send_message failed (routing=%s): %s", routing, exc)
        logging.getLogger("dialog").info(
            "TG_SEND_FAIL chat_id=%s routing=%s exc=%s",
            chat_id,
            routing,
            exc,
        )
        msg = await bot.send_message(
            chat_id,
            text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            **fallback_routing,
        )
        return msg, fallback_routing


class MessageProgress:
    def __init__(
        self,
        message: Message,
        *,
        enabled: bool = True,
        min_interval_s: float = 0.9,
        max_lines: int = 18,
        max_chars: int = 3900,
    ) -> None:
        self._message = message
        self.enabled = enabled
        self.min_interval_s = min_interval_s
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._lines: list[str] = []
        self._last_sent_ts = 0.0
        self._last_text = ""

    async def set(self, text: str) -> None:
        if not self.enabled:
            return
        self._lines = [text]
        await self.flush(force=True)

    async def add(self, line: str) -> None:
        if not self.enabled:
            return
        self._lines.append(line)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]
        await self.flush()

    async def flush(self, *, force: bool = False) -> None:
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and (now - self._last_sent_ts) < self.min_interval_s:
            return

        text = "\n".join(self._lines).strip()
        if not text:
            text = "…"
        if len(text) > self.max_chars:
            text = "…\n" + text[-self.max_chars + 2 :]

        if text == self._last_text and not force:
            return

        try:
            await self._message.edit_text(text[:_TG_MAX_LEN], parse_mode=None)
        except TelegramBadRequest:
            self.enabled = False
            return

        self._last_text = text
        self._last_sent_ts = now


async def _typing_loop(
    bot: Bot,
    chat_id: int,
    stop: asyncio.Event,
    message_thread_id: int | None = None,
) -> None:
    while not stop.is_set():
        with suppress(Exception):
            await bot.send_chat_action(
                chat_id,
                ChatAction.TYPING,
                message_thread_id=message_thread_id,
            )
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

    tg_api = TelegramAPI(settings.telegram.token, proxy_url=settings.telegram.proxy_url)
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

    bot = Bot(token=settings.telegram.token, session=AiohttpSession(proxy=settings.telegram.proxy_url or None))
    dp = Dispatcher()

    pending_topics: dict[int, tuple[dict[str, int], float]] = {}

    def consume_pending_routing(chat_id: int) -> dict[str, int] | None:
        entry = pending_topics.get(chat_id)
        if not entry:
            return None
        routing, ts = entry
        if (time.monotonic() - ts) > _PENDING_TOPIC_TTL_S:
            pending_topics.pop(chat_id, None)
            return None
        pending_topics.pop(chat_id, None)
        return dict(routing)

    @dp.message(F.forum_topic_created)
    async def on_forum_topic_created(message: Message) -> None:
        routing: dict[str, int] = {}
        topic_type = "unknown"
        if message.message_thread_id is not None:
            routing = {"message_thread_id": int(message.message_thread_id)}
            topic_type = "forum"
        else:
            dm_tid = getattr(getattr(message, "direct_messages_topic", None), "topic_id", None)
            if dm_tid is not None:
                routing = {"direct_messages_topic_id": int(dm_tid)}
                topic_type = "direct"

        if routing:
            pending_topics[int(message.chat.id)] = (routing, time.monotonic())
        dialog_logger.info(
            (
                "FORUM_TOPIC_CREATED chat_id=%s msg_id=%s message_thread_id=%s "
                "dm_topic_id=%s topic_type=%s routing=%s title=%s"
            ),
            message.chat.id,
            message.message_id,
            message.message_thread_id,
            getattr(getattr(message, "direct_messages_topic", None), "topic_id", None),
            topic_type,
            routing,
            getattr(getattr(message, "forum_topic_created", None), "name", None),
        )

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        user = db.get_or_create_user(message.from_user.id)
        topics_status = "включены" if topics_enabled else "выключены"
        streaming_status = (
            "включено" if (topics_enabled and settings.telegram.enable_drafts) else "выключено"
        )
        _, routing = _topic_ctx(message)
        await bot.send_message(
            message.chat.id,
            (
                f"Привет! Я бот ВкусВилл. Город: {user.city}.\n"
                f"Темы (forum mode) в личке: {topics_status}.\n"
                f"Стриминг через sendMessageDraft: {streaming_status}.\n\n"
                "Напишите запрос, например: 'молоко'."
            ),
            disable_web_page_preview=True,
            **routing,
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        _, routing = _topic_ctx(message)
        await bot.send_message(
            message.chat.id,
            "Команды:\n"
            "/diet — задать особенности питания\n"
            "/city — задать город\n"
            "Примеры: 'найди молоко', 'состав творога', 'собери корзину: хлеб молоко'",
            disable_web_page_preview=True,
            **routing,
        )

    @dp.message(Command("diet"))
    async def cmd_diet(message: Message) -> None:
        _, routing = _topic_ctx(message)
        text = message.text.replace("/diet", "", 1).strip()
        if not text:
            await bot.send_message(
                message.chat.id,
                "Напишите особенности питания после команды /diet",
                disable_web_page_preview=True,
                **routing,
            )
            return
        db.update_user_diet_notes(message.from_user.id, text)
        await bot.send_message(
            message.chat.id,
            "Сохранил особенности питания.",
            disable_web_page_preview=True,
            **routing,
        )

    @dp.message(Command("city"))
    async def cmd_city(message: Message) -> None:
        _, routing = _topic_ctx(message)
        text = message.text.replace("/city", "", 1).strip()
        if not text:
            await bot.send_message(
                message.chat.id,
                "Напишите город после команды /city",
                disable_web_page_preview=True,
                **routing,
            )
            return
        db.update_user_city(message.from_user.id, text)
        await bot.send_message(
            message.chat.id,
            f"Город обновлён: {text}",
            disable_web_page_preview=True,
            **routing,
        )

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        pending_routing = None
        if message.message_thread_id is None:
            pending_routing = consume_pending_routing(int(message.chat.id))

        thread_key, reply_kwargs = _topic_ctx(message, pending_routing=pending_routing)
        reply_kwargs = dict(reply_kwargs)
        fallback_routing = {"reply_to_message_id": _reply_root_message_id(message)}
        # sendMessageDraft поддерживает только message_thread_id.
        use_drafts = bool(
            settings.telegram.enable_drafts
            and topics_enabled
            and message.message_thread_id is not None
        )
        draft: DraftProgress | None = None
        if use_drafts:
            draft = DraftProgress(
                api=tg_api,
                chat_id=cast(int, message.chat.id),
                draft_id=cast(int, message.message_id),
                message_thread_id=int(message.message_thread_id),
                enabled=bool(settings.telegram.show_progress),
            )
            try:
                await draft.set("🧠 Думаю…\n(прогресс появится ниже)")
            except TelegramAPIError as exc:
                logger.warning("sendMessageDraft недоступен, откатываюсь на fallback: %s", exc)
                draft = None
                use_drafts = False

        placeholder: Message | None = None
        fallback_progress: MessageProgress | None = None
        stop_typing = asyncio.Event()
        typing_task: asyncio.Task[None] | None = None
        if not use_drafts:
            # В некоторых режимах Telegram "угадывает" id топика не так, как мы.
            # Если routing некорректен, падаем на безопасный reply.
            try:
                placeholder, used_routing = await _safe_send(
                    bot,
                    chat_id=cast(int, message.chat.id),
                    text="Думаю…",
                    routing=reply_kwargs,
                    fallback_routing=fallback_routing,
                    parse_mode=None,
                )
                if used_routing != reply_kwargs:
                    reply_kwargs = dict(used_routing)
                    if "reply_to_message_id" in used_routing:
                        thread_key = int(used_routing["reply_to_message_id"])
            except TelegramBadRequest:
                # Если даже fallback не сработал — пусть исключение уйдёт в общий handler.
                raise

            fallback_progress = MessageProgress(
                placeholder,
                enabled=bool(settings.telegram.show_progress),
            )
            await fallback_progress.set("🧠 Думаю…\n(прогресс появится ниже)")
            typing_task = asyncio.create_task(
                _typing_loop(
                    bot,
                    message.chat.id,
                    stop_typing,
                    message_thread_id=(placeholder.message_thread_id or message.message_thread_id),
                )
            )

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
            dialog_logger.info(
                (
                    "USER tg_id=%s user_id=%s chat_id=%s msg_id=%s reply_to=%s "
                    "thread_key=%s thread_id=%s dm_topic_id=%s routing=%s: %s"
                ),
                message.from_user.id,
                user.id,
                message.chat.id,
                message.message_id,
                getattr(getattr(message, "reply_to_message", None), "message_id", None),
                thread_key,
                message.message_thread_id,
                getattr(getattr(message, "direct_messages_topic", None), "topic_id", None),
                reply_kwargs,
                text,
            )
            history = db.get_recent_messages(
                user.id,
                limit=settings.sgr.history_messages,
                thread_id=thread_key,
            )
            progress_cb = draft.add if draft else None
            if not progress_cb and fallback_progress:
                progress_cb = fallback_progress.add
            reply = await agent.run(text, history=history, user_id=user.id, progress=progress_cb)
            db.save_message(user.id, "user", text, thread_id=thread_key)
            db.save_message(user.id, "assistant", reply, thread_id=thread_key)
            dialog_logger.info(
                (
                    "ASSISTANT user_id=%s thread_key=%s thread_id=%s dm_topic_id=%s "
                    "routing=%s: %s"
                ),
                user.id,
                thread_key,
                message.message_thread_id,
                getattr(getattr(message, "direct_messages_topic", None), "topic_id", None),
                reply_kwargs,
                reply,
            )
            db.save_session(user.id, "sgr", {"query": text})

            if draft:
                with suppress(Exception):
                    await draft.add("✅ Готово, отправляю ответ…")

            try:
                reply_md = to_telegram_markdown(reply)
                parts_md = _split_text(reply_md)
                if len(parts_md) > 1:
                    _, used_routing = await _safe_send(
                        bot,
                        chat_id=cast(int, message.chat.id),
                        text="Ответ слишком длинный — отправляю частями.",
                        routing=reply_kwargs,
                        fallback_routing=fallback_routing,
                        parse_mode=None,
                    )
                    reply_kwargs = dict(used_routing)
                for part_md in parts_md:
                    _, used_routing = await _safe_send(
                        bot,
                        chat_id=cast(int, message.chat.id),
                        text=part_md,
                        routing=reply_kwargs,
                        fallback_routing=fallback_routing,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                    reply_kwargs = dict(used_routing)
            except TelegramBadRequest:
                # На всякий случай: если MarkdownV2 не отправился (лимиты/парсинг),
                # отправляем plain-текст частями без разметки.
                for part in _split_text(reply):
                    _, used_routing = await _safe_send(
                        bot,
                        chat_id=cast(int, message.chat.id),
                        text=part,
                        routing=reply_kwargs,
                        fallback_routing=fallback_routing,
                        parse_mode=None,
                    )
                    reply_kwargs = dict(used_routing)

            if placeholder:
                with suppress(TelegramBadRequest):
                    await placeholder.edit_text("✅ Готово")
        except MCPError as exc:
            logger.exception("MCP error")
            if placeholder:
                await placeholder.edit_text(f"Ошибка MCP: {exc}")
            else:
                _, used_routing = await _safe_send(
                    bot,
                    chat_id=cast(int, message.chat.id),
                    text=f"Ошибка MCP: {exc}",
                    routing=reply_kwargs,
                    fallback_routing=fallback_routing,
                    parse_mode=None,
                )
                reply_kwargs = dict(used_routing)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error")
            if placeholder:
                await placeholder.edit_text(f"Ошибка: {exc}")
            else:
                _, used_routing = await _safe_send(
                    bot,
                    chat_id=cast(int, message.chat.id),
                    text=f"Ошибка: {exc}",
                    routing=reply_kwargs,
                    fallback_routing=fallback_routing,
                    parse_mode=None,
                )
                reply_kwargs = dict(used_routing)
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
