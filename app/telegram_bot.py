"""
Telegram Bot 接入层

使用 python-telegram-bot 库构建长连接服务（Polling 模式），
将用户的文本和图片输入传递给 Agent，并回复诊断结果。

Phase 1: 基本文本/图片交互
Phase 2: 多轮记忆 + 图片标注回复
Phase 3: 多专家 Supervisor 架构
"""

import logging
import os
import tempfile

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 最大消息长度（Telegram 限制 4096 字符）
MAX_MESSAGE_LENGTH = 4000


def _split_long_message(text: str) -> list[str]:
    """将长文本拆分为多条消息"""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    parts = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            parts.append(text)
            break
        # 尝试在换行处拆分
        split_pos = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1:
            split_pos = MAX_MESSAGE_LENGTH
        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return parts


# ─────────────────────────────────────────
# 命令处理器
# ─────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    welcome = (
        "🏥 **欢迎使用 MedAgent 智能医学诊断助手！**\n\n"
        "我可以帮你分析CT影像，能力包括：\n"
        "🔬 CT疾病分类（肺炎、肺结节、肺癌等）\n"
        "📍 病灶检测与定位\n"
        "📚 医学知识查询\n"
        "📋 诊断报告生成\n\n"
        "**使用方法：**\n"
        "1️⃣ 直接发送CT图片，我会自动分析\n"
        "2️⃣ 发送文字描述你的需求\n"
        "3️⃣ 发送图片+文字说明，获得更精准的分析\n\n"
        "⚠️ 提示：所有诊断结果仅供参考，请务必就医咨询专业医师。"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = (
        "📖 **MedAgent 使用帮助**\n\n"
        "/start - 开始使用\n"
        "/help  - 查看帮助\n\n"
        "**发送图片**：我会对CT图片进行疾病分类和病灶检测\n"
        "**发送文字**：可以提问医学问题或描述症状\n"
        "**图片+文字**：添加图片说明，获得更有针对性的分析\n\n"
        "💡 支持多轮对话，你可以追问之前的分析结果。"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ─────────────────────────────────────────
# 消息处理器
# ─────────────────────────────────────────

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户发送的纯文本消息"""
    user_text = update.message.text
    user_id = str(update.effective_user.id)

    logger.info("收到文本消息 [user=%s]: %s", user_id, user_text[:50])

    await update.message.reply_text("🔍 正在分析你的问题，请稍候...")

    try:
        from app.agent.orchestrator import run_supervisor_agent

        reply = run_supervisor_agent(
            message=user_text,
            thread_id=user_id,
        )

        # 发送回复（可能需要拆分长消息）
        for part in _split_long_message(reply):
            await update.message.reply_text(part, parse_mode="Markdown")

    except Exception as exc:
        logger.error("处理文本消息失败: %s", exc)
        await update.message.reply_text(
            f"❌ 抱歉，处理过程中出现错误: {str(exc)[:200]}\n请稍后重试。"
        )


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户发送的图片（CT图片）"""
    user_id = str(update.effective_user.id)

    # 获取图片附带的文字说明
    caption = update.message.caption or "请分析这张CT图片"

    logger.info("收到图片消息 [user=%s], caption: %s", user_id, caption[:50])

    await update.message.reply_text("📸 已收到CT图片，正在分析中，请稍候...")

    tmp_path = None
    annotated_path = None
    try:
        # 下载图片到临时文件
        photo = update.message.photo[-1]  # 获取最大尺寸的图片
        photo_file = await photo.get_file()

        suffix = ".jpg"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="medagent_ct_")
        os.close(tmp_fd)

        await photo_file.download_to_drive(tmp_path)
        logger.info("图片已保存至: %s", tmp_path)

        # 调用 Supervisor Agent 进行分析
        from app.agent.orchestrator import run_supervisor_agent

        reply = run_supervisor_agent(
            message=caption,
            thread_id=user_id,
            image_path=tmp_path,
        )

        # 尝试生成标注图片
        try:
            annotated_path = _try_annotate_image(tmp_path, reply)
        except Exception as ann_exc:
            logger.warning("图片标注失败（非致命）: %s", ann_exc)

        # 发送标注图片（如果有）
        if annotated_path and os.path.exists(annotated_path):
            with open(annotated_path, "rb") as img_file:
                await update.message.reply_photo(
                    photo=img_file,
                    caption="🔍 病灶检测标注图",
                )

        # 发送文本报告
        for part in _split_long_message(reply):
            await update.message.reply_text(part, parse_mode="Markdown")

    except Exception as exc:
        logger.error("处理图片消息失败: %s", exc)
        await update.message.reply_text(
            f"❌ 抱歉，图片分析过程中出现错误: {str(exc)[:200]}\n请稍后重试。"
        )
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if annotated_path and os.path.exists(annotated_path):
            try:
                os.unlink(annotated_path)
            except OSError:
                pass


def _try_annotate_image(image_path: str, agent_reply: str) -> str | None:
    """
    尝试从 Agent 回复中提取检测结果并标注图片

    返回标注后的图片路径，如果无法标注则返回 None
    """
    try:
        import json
        from app.skills.image_annotator import annotate_ct_image
        from app.skills.lesion_detector import detect_lesions

        # 调用检测器获取坐标数据
        detection_result = detect_lesions(image_path)
        lesions = detection_result.get("lesions", [])

        if not lesions:
            return None

        # 调用标注模块
        annotated_path = annotate_ct_image(image_path, lesions)
        return annotated_path

    except ImportError:
        logger.debug("image_annotator 模块尚未安装，跳过标注")
        return None
    except Exception as exc:
        logger.warning("图片标注过程出错: %s", exc)
        return None


# ─────────────────────────────────────────
# Bot 启动入口
# ─────────────────────────────────────────

def main() -> None:
    """启动 Telegram Bot（Polling 模式）"""
    token = settings.telegram_bot_token
    if not token:
        logger.error(
            "未配置 TELEGRAM_BOT_TOKEN！"
            "请在 .env 文件中添加: TELEGRAM_BOT_TOKEN=your_token_here"
        )
        return

    logger.info("正在启动 MedAgent Telegram Bot...")

    # 构建 Application
    app = Application.builder().token(token).build()

    # 注册命令处理器
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # 注册消息处理器
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    # 启动 Polling
    logger.info("MedAgent Bot 已启动，等待消息...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
