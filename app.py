import io
import os
import time
import tempfile
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI

# -----------------------------
# 初期化
# -----------------------------
load_dotenv()
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    st.warning("OPENAI_API_KEY が設定されていません。 .env に追加してください。")

client = OpenAI()

APP_TITLE = "🇻🇳⇄🇯🇵⇄🇺🇸 多言語翻訳 (テキスト + 音声)"
STT_MODEL = "gpt-4o-mini-transcribe"     # 音声→テキスト
TTS_MODEL = "gpt-4o-mini-tts"             # テキスト→音声
LLM_MODEL = "gpt-4o-mini"                 # 翻訳

# Mobile-friendly: wide layout collapses sidebar by default on phones
st.set_page_config(page_title=APP_TITLE, page_icon="🌏", layout="wide", initial_sidebar_state="collapsed")
st.title(APP_TITLE)
st.caption("テキスト翻訳、マイク入力、音声会話。Streamlit + OpenAI で構築。")

# Keep language choices in session and provide a one-click swap
if "src" not in st.session_state:
    st.session_state.src = "vi"
if "dst" not in st.session_state:
    st.session_state.dst = "ja"

def swap_langs():
    st.session_state.src, st.session_state.dst = st.session_state.dst, st.session_state.src

# -----------------------------
# ヘルパー関数
# -----------------------------

def detect_lang_simple(text: str) -> str:
    """ベトナム語/日本語/英語の簡易判定"""
    if any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text):
        return "ja"
    try:
        lang = detect(text)
        if lang in ("ja", "vi", "en"):
            return lang
    except Exception:
        pass
    # Simple heuristic: if mostly ASCII, likely English or Vietnamese
    if all(ord(c) < 128 for c in text):
        # Basic check for English vs Vietnamese
        english_words = ["the", "and", "is", "are", "was", "were", "have", "has", "will", "would", "can", "could"]
        vietnamese_chars = ["ă", "â", "đ", "ê", "ô", "ơ", "ư", "á", "à", "ả", "ã", "ạ"]
        
        text_lower = text.lower()
        has_english = any(word in text_lower for word in english_words)
        has_vietnamese = any(char in text_lower for char in vietnamese_chars)
        
        if has_vietnamese:
            return "vi"
        elif has_english:
            return "en"
        else:
            return "vi"  # default fallback
    return "ja"


def detect_formality_and_context(text: str, lang: str) -> dict:
    """AI-powered formality and context detection"""
    analysis_prompt = f"""
    Analyze the following text in {lang} language and determine:
    1. Formality level: casual, neutral, formal, very_formal
    2. Context: personal, business, academic, technical, creative, medical, legal
    3. Tone: friendly, professional, serious, playful, urgent, polite
    
    Text: "{text}"
    
    Respond with only a JSON object like:
    {{"formality": "formal", "context": "business", "tone": "professional"}}
    """
    
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        import json
        result = json.loads(resp.choices[0].message.content.strip())
        return result
    except Exception:
        # Fallback to simple heuristics
        text_lower = text.lower()
        
        # Simple formality detection
        formal_indicators = ["please", "thank you", "sincerely", "respectfully", "でございます", "いたします", "xin chào", "kính chào"]
        casual_indicators = ["hey", "yo", "だよ", "だね", "ね", "よ", "chào bạn", "ơi"]
        
        if any(indicator in text_lower for indicator in formal_indicators):
            formality = "formal"
        elif any(indicator in text_lower for indicator in casual_indicators):
            formality = "casual"
        else:
            formality = "neutral"
            
        return {"formality": formality, "context": "personal", "tone": "friendly"}


def translate_text(text: str, src: str, dst: str) -> str:
    if src == "auto":
        detected = detect_lang_simple(text)
        if detected in ("vi", "ja", "en"):
            src = detected
        else:
            src = "vi"  # default fallback
    if src == dst:
        return text

    # AI-powered context analysis
    context_info = detect_formality_and_context(text, src)
    formality = context_info.get("formality", "neutral")
    context = context_info.get("context", "personal")
    tone = context_info.get("tone", "friendly")

    # Create context-aware system prompt
    base_prompt = "あなたはプロの翻訳者です。"
    
    if formality == "very_formal":
        style_instruction = "最も丁寧で格式高い表現を使用し、敬語を適切に使い分けてください。"
    elif formality == "formal":
        style_instruction = "丁寧で正式な表現を使用し、ビジネス文書や公式な場面に適した翻訳をしてください。"
    elif formality == "casual":
        style_instruction = "自然でカジュアルな表現を使用し、日常会話に適した親しみやすい翻訳をしてください。"
    else:  # neutral
        style_instruction = "自然で適度に丁寧な表現を使用してください。"
    
    if context == "business":
        context_instruction = "ビジネス文書として適切な専門用語と表現を使用してください。"
    elif context == "academic":
        context_instruction = "学術的で正確な表現を使用し、専門性を保ってください。"
    elif context == "technical":
        context_instruction = "技術的な内容として正確性を重視し、専門用語を適切に翻訳してください。"
    elif context == "medical":
        context_instruction = "医療用語を正確に翻訳し、専門性と正確性を最優先してください。"
    elif context == "legal":
        context_instruction = "法的文書として正確で曖昧さのない表現を使用してください。"
    else:  # personal, creative
        context_instruction = "感情やニュアンスを大切にし、人間味のある自然な表現を心がけてください。"

    system_prompt = f"""
    {base_prompt}
    
    翻訳スタイル: {style_instruction}
    文脈考慮: {context_instruction}
    
    - ソース言語: 'vi'=ベトナム語, 'ja'=日本語, 'en'=英語
    - ターゲット言語: 'ja'=日本語, 'vi'=ベトナム語, 'en'=英語
    - 検出された調子: {tone}
    - 文脈: {context}
    - 丁寧度: {formality}
    
    元のテキストの調子と文脈を保ちながら、上記スタイルで翻訳してください。
    数字や名前はそのまま保持し、説明は追加せず翻訳文のみ出力してください。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[SRC={src}] [DST={dst}]\n{text}"},
    ]

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,  # type: ignore
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip() if resp.choices[0].message.content else "Translation failed"
    except Exception as e:
        return f"Translation error: {str(e)}"


def transcribe_bytes(wav_bytes: bytes, lang_hint: str = "auto") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            kwargs = {"model": STT_MODEL, "file": f}
            if lang_hint in ("vi", "ja", "en"):
                kwargs["language"] = lang_hint
            stt = client.audio.transcriptions.create(**kwargs)
        return stt.text.strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def speak(text: str, voice: str = "alloy", fmt: str = "mp3"):
    """TTS -> bytes. No local conversion; we ask the API for mp3 or wav."""
    if not text.strip():
        return b"", "audio/mp3"

    params = {
        "model": TTS_MODEL,
        "voice": voice,
        "input": text,
    }
    # Ask the API for the format we want
    if fmt in ("mp3", "wav"):
        params["response_format"] = fmt

    try:
        resp = client.audio.speech.create(**params)
        audio_bytes = resp.read()
        mime = "audio/mp3" if fmt == "mp3" else "audio/wav"
        return audio_bytes, mime
    except Exception as e:
        st.error(f"TTS error: {e}")
        return b"", "audio/mp3"


# -----------------------------
# UI サイドバー
# -----------------------------
with st.sidebar:
    st.header("⚙️ 設定 / Cài đặt")
    mode = st.radio("モード / Chế độ", ["テキスト翻訳 / Dịch văn bản", "音声入力 / Ghi âm", "会話モード / Hội thoại"], index=0) or "テキスト翻訳 / Dịch văn bản"
    st.divider()

    st.subheader("翻訳設定 / Cấu hình dịch")
    # language row: [src] [⇄] [dst]
    col1, colS, col2 = st.columns([1, 0.25, 1])
    with col1:
        st.selectbox("入力言語 / Ngôn ngữ nguồn", ["vi", "ja", "en"], key="src")
    with colS:
        st.button("⇄", help="入力/出力を入れ替え · Đổi chiều", on_click=swap_langs, use_container_width=True)
    with col2:
        st.selectbox("出力言語 / Ngôn ngữ đích", ["ja", "vi", "en"], key="dst")

    st.divider()
    st.subheader("音声設定 / Cấu hình giọng nói")
    tts_voice = st.selectbox("音声タイプ / Giọng", ["alloy", "verse", "aria", "sage"], index=0) or "alloy"
    audio_format = st.selectbox("音声形式 / Định dạng", ["mp3", "wav"], index=1) or "wav"

# read current choices from session
src_choice = st.session_state.src
dst_choice = st.session_state.dst

# -----------------------------
# 各モード (UI 表示も日越併記)
# -----------------------------
if mode.startswith("テキスト"):
    st.subheader("📝 テキスト翻訳 / Dịch văn bản")
    if dst_choice == "ja":
        example = "Xin chào, rất vui được gặp bạn." if src_choice == "vi" else "Hello, nice to meet you."
    elif dst_choice == "vi":
        example = "今日はとても暑いですね。" if src_choice == "ja" else "The weather is very hot today."
    else:  # dst_choice == "en"
        example = "今日はとても暑いですね。" if src_choice == "ja" else "Xin chào, rất vui được gặp bạn."
    
    text_in = st.text_area("テキスト入力 / Nhập văn bản", example, height=150)
    if st.button("翻訳 / Dịch", type="primary"):
        if not text_in.strip():
            st.warning("テキストを入力してください / Vui lòng nhập văn bản")
        else:
            with st.spinner("AI分析中... / Đang phân tích AI..."):
                # First, detect context and formality
                context_info = detect_formality_and_context(text_in, src_choice)
                
                # Show AI analysis
                with st.expander("🤖 AI分析結果 / Kết quả phân tích AI", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        formality_emoji = {"casual": "😊", "neutral": "😐", "formal": "🎩", "very_formal": "👔"}
                        formality_jp = {"casual": "カジュアル", "neutral": "普通", "formal": "丁寧", "very_formal": "非常に丁寧"}
                        current_formality = context_info.get("formality", "neutral")
                        st.metric("丁寧度 / Độ lịch sự / Formality", 
                                f"{formality_jp.get(current_formality, '普通')} / {current_formality}", 
                                delta=f"{formality_emoji.get(current_formality, '😐')}")
                    with col2:
                        context_emoji = {"personal": "👥", "business": "💼", "academic": "🎓", "technical": "⚙️", "creative": "🎨", "medical": "🏥", "legal": "⚖️"}
                        context_jp = {"personal": "個人的", "business": "ビジネス", "academic": "学術的", "technical": "技術的", "creative": "創作的", "medical": "医療", "legal": "法的"}
                        current_context = context_info.get("context", "personal")
                        st.metric("文脈 / Ngữ cảnh / Context", 
                                f"{context_jp.get(current_context, '個人的')} / {current_context}",
                                delta=f"{context_emoji.get(current_context, '👥')}")
                    with col3:
                        tone_emoji = {"friendly": "😊", "professional": "💼", "serious": "😐", "playful": "😄", "urgent": "⚡", "polite": "🙏"}
                        tone_jp = {"friendly": "親しみやすい", "professional": "プロ的", "serious": "真面目", "playful": "遊び心", "urgent": "緊急", "polite": "礼儀正しい"}
                        current_tone = context_info.get("tone", "friendly")
                        st.metric("調子 / Giọng điệu / Tone", 
                                f"{tone_jp.get(current_tone, '親しみやすい')} / {current_tone}",
                                delta=f"{tone_emoji.get(current_tone, '😊')}")
                
            with st.spinner("翻訳中... / Đang dịch..."):
                out = translate_text(text_in, src_choice, dst_choice)
            st.success("完了 / Hoàn tất")
            st.markdown("**翻訳結果 / Kết quả**")
            st.text_area("", out, height=150)
            audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
            if audio_bytes:
                st.audio(audio_bytes, format=mime)

elif mode.startswith("音声入力"):
    st.subheader("🎤 音声入力翻訳 / Dịch giọng nói")
    st.caption("クリックして録音 / Nhấn để ghi âm")

    # Smaller icon for phones
    wav_bytes = audio_recorder(text="録音 / Ghi âm", recording_color="#e53935", neutral_color="#6c757d", icon_size="1.6x")
    if wav_bytes:
        st.info("録音完了 / Đã ghi âm. テキスト化中... / Đang nhận dạng...")
        transcript = transcribe_bytes(wav_bytes, src_choice if src_choice != "auto" else "auto")
        st.markdown("**文字起こし / Văn bản**")
        st.markdown(f"<div style='font-size: 1.5em; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 10px 0; color: #333333;'>{transcript}</div>", unsafe_allow_html=True)

        # AI Context Analysis
        with st.spinner("AI分析中... / Đang phân tích AI..."):
            context_info = detect_formality_and_context(transcript, src_choice)
            
        with st.expander("🤖 AI分析結果 / Kết quả phân tích AI", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                formality_emoji = {"casual": "😊", "neutral": "😐", "formal": "🎩", "very_formal": "👔"}
                formality_jp = {"casual": "カジュアル", "neutral": "普通", "formal": "丁寧", "very_formal": "非常に丁寧"}
                current_formality = context_info.get("formality", "neutral")
                st.metric("丁寧度 / Độ lịch sự / Formality", 
                        f"{formality_jp.get(current_formality, '普通')} / {current_formality}", 
                        delta=f"{formality_emoji.get(current_formality, '😐')}")
            with col2:
                context_emoji = {"personal": "👥", "business": "💼", "academic": "🎓", "technical": "⚙️", "creative": "🎨", "medical": "🏥", "legal": "⚖️"}
                context_jp = {"personal": "個人的", "business": "ビジネス", "academic": "学術的", "technical": "技術的", "creative": "創作的", "medical": "医療", "legal": "法的"}
                current_context = context_info.get("context", "personal")
                st.metric("文脈 / Ngữ cảnh / Context", 
                        f"{context_jp.get(current_context, '個人的')} / {current_context}",
                        delta=f"{context_emoji.get(current_context, '👥')}")
            with col3:
                tone_emoji = {"friendly": "😊", "professional": "💼", "serious": "😐", "playful": "😄", "urgent": "⚡", "polite": "🙏"}
                tone_jp = {"friendly": "親しみやすい", "professional": "プロ的", "serious": "真面目", "playful": "遊び心", "urgent": "緊急", "polite": "礼儀正しい"}
                current_tone = context_info.get("tone", "friendly")
                st.metric("調子 / Giọng điệu / Tone", 
                        f"{tone_jp.get(current_tone, '親しみやすい')} / {current_tone}",
                        delta=f"{tone_emoji.get(current_tone, '😊')}")

        with st.spinner("翻訳中... / Đang dịch..."):
            out = translate_text(transcript, src_choice, dst_choice)
        st.markdown("**翻訳 / Bản dịch**")
        st.markdown(f"<div style='font-size: 1.7em; font-weight: bold; padding: 15px; background-color: #e8f4fd; border-radius: 5px; margin: 10px 0; border-left: 4px solid #1f77b4; color: #1f77b4;'>{out}</div>", unsafe_allow_html=True)

        audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
        if audio_bytes:
            st.audio(audio_bytes, format=mime)

elif mode.startswith("会話"):
    st.subheader("🗣️ 会話モード / Hội thoại")
    st.caption("交互に話してください / Nói lần lượt")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Smaller icon for phones
    wav_bytes = audio_recorder(text="話す / Nói", recording_color="#1e88e5", neutral_color="#6c757d", icon_size="1.6x")
    if wav_bytes:
        transcript = transcribe_bytes(wav_bytes, "auto")
        detected = detect_lang_simple(transcript)
        
        # Vice versa translation based on translation settings
        # If detected language matches source setting, translate to destination
        # If detected language matches destination setting, translate to source
        if detected == src_choice:
            target = dst_choice
        elif detected == dst_choice:
            target = src_choice
        else:
            # If detected language doesn't match either setting, use default logic
            if detected == "vi":
                target = "ja"
            elif detected == "ja":
                target = "vi" 
            elif detected == "en":
                target = dst_choice if dst_choice != "en" else "ja"
            else:
                target = dst_choice
            
        translation = translate_text(transcript, detected, target)
        st.session_state.chat.append({
            "speaker": "A" if (len(st.session_state.chat) % 2 == 0) else "B",
            "transcript": transcript,
            "translation": translation,
            "src": detected,
            "dst": target,
        })
        audio_bytes, mime = speak(translation, voice=tts_voice, fmt=audio_format)
        if audio_bytes:
            st.audio(audio_bytes, format=mime)

    for i, msg in enumerate(reversed(st.session_state.chat)):
        role = msg["speaker"]
        st.markdown(f"### {len(st.session_state.chat)-i} 回目 / Lượt {len(st.session_state.chat)-i} · 話者 / Người nói {role}")
        
        # Original text with larger font
        st.markdown("**原文 / Văn bản gốc:**")
        st.markdown(f"<div style='font-size: 1.4em; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 5px 0; color: #333333;'><em>({msg['src']})</em> {msg['transcript']}</div>", unsafe_allow_html=True)
        
        # Translation with larger, more prominent font
        st.markdown("**翻訳 / Bản dịch:**")
        st.markdown(f"<div style='font-size: 1.6em; font-weight: bold; padding: 15px; background-color: #e8f4fd; border-radius: 5px; margin: 5px 0; border-left: 4px solid #1f77b4; color: #1f77b4;'><em>({msg['dst']})</em> {msg['translation']}</div>", unsafe_allow_html=True)
        
        st.divider()

# -----------------------------
# Footer
# -----------------------------
st.caption("🤖 AI-Powered Context-Aware Translation · コンテキスト認識AI翻訳 · Dịch thuật AI nhận thức ngữ cảnh")
st.caption("❤️ Streamlit + OpenAI で構築 · Xây dựng bằng Streamlit và OpenAI · FFmpeg 推奨 / Nên cài FFmpeg")

