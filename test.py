import streamlit as st
import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
from translate import Translator
#from paddlespeech.cli.tts.infer import TTSExecutor
import io
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
import paddle.nn.functional as F
import paddle
import matplotlib.pyplot as plt
#iort numpy as np
import edge_tts
import asyncio
import numpy as np
from nltk.tokenize import sent_tokenize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# 使用 edge-tts 进行中文语音合成
async def speak_text_zh(text):
    try:
        communicate = edge_tts.Communicate(text, voice='zh-CN-XiaoxiaoNeural')
        await communicate.save('output_zh.wav')
        audio_file = open('output_zh.wav', 'rb')
        st.audio(audio_file, format='audio/wav')
    except Exception as e:
        st.error(f"Error generating Chinese speech: {e}")

# 使用 edge-tts 进行英文语音合成
async def speak_text_en(text):
    try:
        communicate = edge_tts.Communicate(text, voice='en-US-JennyNeural')
        await communicate.save('output_en.wav')
        audio_file = open('output_en.wav', 'rb')
        st.audio(audio_file, format='audio/wav')
    except Exception as e:
        st.error(f"Error generating English speech: {e}")


# 使用 translate 库进行翻译
def translate_text(text, from_lang='en', to_lang='zh'):
    try:
        translator = Translator(from_lang=from_lang, to_lang=to_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return ""
    
# 替换现有的 translate_sentences 函数
def translate_sentences(sentences, from_lang="en", to_lang="zh"):
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    translations = []
    for sentence in sentences:
        try:
            translation = translator.translate(sentence)
            translations.append(translation)
        except Exception as e:
            st.error(f"Error translating sentence: {e}")
            translations.append(f"Translation error for: {sentence}")
    return translations



# 获取单词定义
def get_definitions(word):
    synonyms = wordnet.synsets(word)
    definitions = []
    for synset in synonyms:
        definitions.append(synset.definition())
    return definitions[:5]  # 只获取前10个最常用的定义

# 获取例句
def get_examples(word):
    example_sentences = []
    for file_id in reuters.fileids():  # 使用直接导入的 reuters
        raw_text = reuters.raw(file_id)  # 获取整个文件的原始文本
        sentences = sent_tokenize(raw_text)  # 使用 NLTK 的 sent_tokenize 进行分句处理
        
        # 遍历每个句子，检查是否包含目标单词
        for sentence in sentences:
            if word.lower() in sentence.lower():
                example_sentences.append(sentence)
    
    return example_sentences[:5]  # 只显示最多5个例句

# 判断输入语言是否为中文
def is_chinese(text):
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

# Streamlit App Layout
title_template = """
<div style="background-color:darkblue; padding:20px; border-radius:10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
    <h1 style="color:pink; text-align:center; font-family:Arial, sans-serif;">基于情感分析驱动下的生词学习应用</h1>
</div>
"""
st.markdown(title_template, unsafe_allow_html=True)

subtitle_template = """
<div style="background-color:lightblue; padding:20px; border-radius:10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
    <h1 style="color:pink; text-align:center; font-family:Arial, sans-serif;">本应用由Streamlit和PaddlePaddle相关模型构建</h1>
</div>
"""
st.markdown(subtitle_template, unsafe_allow_html=True)
#st.sidebar.image('NLP.png', use_column_width=True)

activity = ['单词的可视化情感分析与发音指导','关于我们']
choice = st.sidebar.selectbox("菜单", activity)


if choice == '单词的可视化情感分析与发音指导':
    user_word = st.text_input("请输入一个单词:")

    if user_word:
        with st.spinner("获取定义、翻译和示例……"):

            # 获取定义
            definitions = get_definitions(user_word)
            if definitions:
                st.subheader(f'"{user_word}"的英文定义:')
                for i, definition in enumerate(definitions):
                    st.write(f"{i+1}. {definition}")

                # 显示定义的中文翻译并进行情感分析可视化
                st.subheader(f'对于 "{user_word}"的定义的中文翻译:')
                chinese_translations = []
                for i, definition in enumerate(definitions):
                    translation = translate_text(definition)
                    if translation:
                        chinese_translations.append(translation)
                        st.write(f"{i+1}. {translation}")


            # 获取例句并显示
            example_sentences = get_examples(user_word)
            if example_sentences:
                st.subheader(f'使用 "{user_word}"的例句:')
                for i, example in enumerate(example_sentences):
                    st.write(f"{i+1}. {example}")


                # 翻译英文例句为中文
                translated_examples = translate_sentences(example_sentences)
                st.subheader(f'对于使用 "{user_word}"例句的中文翻译:')
                for i, translation in enumerate(translated_examples):
                    st.write(f"{i+1}. {translation}")
                


        if definitions or example_sentences:
            st.subheader("发音指导")
            # 添加语音合成功能的按钮等

        # 新的展示栏：可视化情感分析结果
        elif choice == 'Sentiment Analysis Visualization':
            st.subheader("Sentiment Analysis Visualization")

            # 选择要展示的情感分析结果
            visualization_options = [
                'English Word Sentiment',
                'English Definitions Sentiment',
                'Chinese Definitions Sentiment',
                'English Example Sentences Sentiment'
            ]
            
            selected_option = st.selectbox("Select a sentiment analysis to visualize", visualization_options)



        if definitions or example_sentences:
        # 添加 CSS 以设置按钮样式
           st.markdown(
            """
            <style>
            .custom-button {
                background-color: #90EE90; /* 浅绿色 */
                color: #2F4F4F; /* 深灰色字体 */
                border: none;
                padding: 12px;
                width: 100%;
                cursor: pointer;
                transition: background-color 0.3s, color 0.3s; /* 背景颜色和字体颜色变化的过渡 */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* 设置字体 */
                font-size: 16px; /* 字体大小 */
                border-radius: 8px; /* 圆角边框 */
            }
            .custom-button:hover {
                background-color: #006400; /* 深绿色 */
                color: #FFFFFF; /* 白色字体 */
            }
            .button-container {
                display: flex;
                gap: 12px; /* 按钮之间的间距 */
            }
            </style>

        """,
        unsafe_allow_html=True,)
           st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
           if st.button("朗读单词 (英文)", key="btn1", help="点击朗读英文单词"):
               asyncio.run(speak_text_en(user_word))
    
           if st.button("朗读定义(英文)", key="btn2", help="点击朗读英文定义"):
               definitions_text = "\n".join(definitions)
               asyncio.run(speak_text_en(definitions_text))
    
           if st.button("朗读定义(中文)", key="btn3", help="点击朗读中文含义"):
               translations_text = "\n".join(chinese_translations)
               asyncio.run(speak_text_zh(translations_text))
    
           if st.button("朗读例句(英文)", key="btn4", help="点击朗读英文例句"):
               examples_text = "\n".join(example_sentences)
               asyncio.run(speak_text_en(examples_text))
    
           if st.button("朗读例句(中文)", key="btn5", help="点击朗读中文例句"):
               examples_text2 = "\n".join(translated_examples)
               asyncio.run(speak_text_zh(examples_text2))
    
           st.markdown('</div>', unsafe_allow_html=True)



elif choice == '关于我们':
    st.subheader("关于我们")
    templates = """
    <div style="background-color:lightblue; padding:10px; display:flex; flex-wrap:wrap; justify-content:space-between; border-radius:8px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">

   
    <h4 style=" text-align:center;">使用说明<h4>
    <p>在输入框内输入一个英文单词，按enter键，等待片刻，便会出现相关内容，包括对该单词的情感分析，中英文环境不同情景下的情感分析，可视化展示，最后点击四个按钮便可进行中英文发音学习，可根据您的需要进行音频播放速率调节和音频下载<p>
    <p>希望我们的应用能给您带来便利，祝您学习顺利，生活愉快!<p>
    <p>如果有使用上的问题，请联系我们，<br>联系方式：garrygarry03@outlook.com,xsm829@126.com
    <p>
    </div>
    """
    st.markdown(templates, unsafe_allow_html=True)
    st.markdown("""想了解更多信息:
    - [Streamlit](https://docs.streamlit.io/)
    - [PaddlePaddle](https://www.paddlepaddle.org.cn/)""")


# Streamlit 页面
st.title("导出页面为PDF")
st.success("提示：选择右上角的三个点符号，点击后里面有print，就可以在浏览器里面下载整个页面的PDF啦！")