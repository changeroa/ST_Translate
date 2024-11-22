#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install streamlit transformers sentencepiece pyaudio


# In[4]:


import streamlit as st
import numpy as np
import pyaudio
import time
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
from peft import PeftModel


# In[15]:


# 모델 로드
@st.cache_resource
def load_models():
    # 음성인식 모델
    base_model_path = "openai/whisper-base"
    lora_model_path = "lora_whisper_model"
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path)
    asr_model = PeftModel.from_pretrained(base_model, lora_model_path)
    asr_processor = WhisperProcessor.from_pretrained(base_model_path)

    # 번역 모델
    translation_model_name = "Helsinki-NLP/opus-mt-ko-en"
    trans_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    trans_model = MarianMTModel.from_pretrained(translation_model_name)

    return asr_processor, asr_model, trans_tokenizer, trans_model

# 글자 애니메이션
def display_typing(text, delay=0.05):
    placeholder = st.empty()  
    typed_text = ""

    for char in text:
        typed_text += char
        placeholder.text(typed_text) 
        time.sleep(delay)

# 글자 애니메이션 후 삭제
def display_typing_clear(text, delay=0.05):
    placeholder = st.empty()
    typed_text = ""

    # 애니메이션으로 텍스트 출력
    for char in text:
        typed_text += char
        placeholder.text(typed_text)
        time.sleep(delay)

    # placeholder 반환
    return placeholder

# 글자 삭제
placeholder_2 = st.empty()

# 번역 함수
def translate_korean_phrases(text):
    phrases = re.split(r'([a-zA-Z0-9]+)', text)
    translated_phrases = []

    for phrase in phrases:
        if re.search(r'[가-힣]', phrase):  # 한국어 포함된 구문 번역
            inputs = trans_tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
            outputs = trans_model.generate(**inputs)
            translated_phrase = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_phrases.append(translated_phrase)
        else:
            translated_phrases.append(phrase)

    return ' '.join(translated_phrases)

# 녹음 함수
def record_audio_with_progress(duration):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    progress_bar = st.progress(0)  
    start_time = time.time()

    placeholder = display_typing_clear("녹음 중...")

    while time.time() - start_time < duration:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # 진행 바 업데이트
            elapsed_time = time.time() - start_time
            progress = min(100, int((elapsed_time / duration) * 100))
            progress_bar.progress(progress)
        except Exception as e:
            st.error(f"녹음 중 오류 발생! {e}")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    placeholder.empty()
    st.write("녹음 완료!")
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
    return audio_data

# Streamlit 페이지 구성
st.markdown("<hr style='border: 2px solid #778899;'>", unsafe_allow_html=True)
st.markdown(
    """
    <h1 style='text-align: center;'>🤹 <span style='color:#6495ed;'>코드스위칭</span> 발화를 번역해요! 🌀</h1>
    """, 
    unsafe_allow_html=True
)
st.write("")
st.markdown("<p style='font-size:19px; text-align:center; '>한국외국어대학교 데이터 분석 학회 DAT</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px; text-align:center; color:#696969;'>김세린 김예경 이지원 조재표</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #778899;'>", unsafe_allow_html=True) 

placeholder = display_typing_clear("모델 로드 중...")
asr_processor, asr_model, trans_tokenizer, trans_model = load_models()
placeholder.empty()
placeholder_2.write("모델 로드 완료!")
placeholder_2.empty()

# UI 구성
st.write("")
st.header(":loud_sound: 음성 녹음")
st.write("")

duration = st.slider("녹음 시간을 선택해주세요. (sec)", 1, 30, 5)
if st.button("START"):
    # 녹음 시작
    audio_data = record_audio_with_progress(duration)

    st.write("")
    st.divider()
    st.write("")
    st.header(":memo: 영어 번역")
    st.write("")
    st.write("")
    time.sleep(2)
    
    placeholder = display_typing_clear("오디오 데이터 처리 중...")
    audio_data_float = audio_data.astype(np.float32) / 32768.0
    input_features = asr_processor(audio_data_float, sampling_rate=16000, return_tensors="pt").input_features
    placeholder.empty()
    
    placeholder = display_typing_clear("Whisper 모델로 전사 중...")
    predicted_ids = asr_model.generate(input_features)
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    placeholder.empty()

    st.subheader(":twisted_rightwards_arrows: 전사 결과")
    st.markdown(f"<p style='font-size:23px;'>{transcription}</p>", unsafe_allow_html=True)
    time.sleep(2)
    
    placeholder = display_typing_clear("번역 중...")
    translated_text = translate_korean_phrases(transcription)
    placeholder.empty()
    st.write("")

    st.subheader(":abcd: 번역 결과")
    st.markdown(f"<p style='font-size:23px;'>{translated_text}</p>", unsafe_allow_html=True)


# In[ ]:




