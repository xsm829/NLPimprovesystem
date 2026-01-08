import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
from translate import Translator
import io
import edge_tts
import asyncio
import numpy as np
from nltk.tokenize import sent_tokenize

# 使用 translate 库进行翻译
def translate_text(text, from_lang='en', to_lang='zh'):
    try:
        translator = Translator(from_lang=from_lang, to_lang=to_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        raise Exception(f"Error translating text: {e}")
    
# 翻译句子
def translate_sentences(sentences, from_lang="en", to_lang="zh"):
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    translations = []
    for sentence in sentences:
        try:
            translation = translator.translate(sentence)
            translations.append(translation)
        except Exception as e:
            translations.append(f"Translation error for: {sentence}")
    return translations

# 获取单词定义
def get_definitions(word):
    synonyms = wordnet.synsets(word)
    definitions = []
    for synset in synonyms:
        definitions.append(synset.definition())
    return definitions[:5]  # 只获取前5个最常用的定义

# 获取例句
def get_examples(word):
    example_sentences = []
    for file_id in reuters.fileids():
        raw_text = reuters.raw(file_id)
        sentences = sent_tokenize(raw_text)
        
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

# 处理单词学习请求
async def process_word_learning(word):
    output = []

    # 获取定义
    definitions = get_definitions(word)
    if definitions:
        output.append("Definitions:")
        for definition in definitions:
            # 翻译定义
            translation = translate_text(definition)
            if translation:
                output.append(f"{definition} (Translation: {translation})")
            else:
                output.append(definition)

    # 获取例句
    example_sentences = get_examples(word)
    if example_sentences:
        output.append("\nExamples:")
        for sentence in example_sentences:
            # 翻译例句
            translation = translate_text(sentence)
            if translation:
                output.append(f"{sentence} (Translation: {translation})")
            else:
                output.append(sentence)

    return "\n".join(output)

