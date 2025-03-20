import yake
import nltk
import traceback
from rake_nltk import Rake
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer


def download_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


# pega palavras chaves usando yake, solução simples e de baixo custo computacional
def theme_from_yake(query):
    # somente top=3 pq o texto é pode ser muito pequeno
    extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=3)
    return " ".join([keyword for keyword, _ in extractor.extract_keywords(query)])


# o mesmo que o yake, pouco mais custoso e resultado (teoricamente) melhor em extração de palavras-chave
def theme_from_rake(query):
    try:
        download_punkt()  # se já existir, não faz nada
        rake = Rake()
        rake.extract_keywords_from_text(query)
        print(f"len of keyword list = {len(rake.get_ranked_phrases())}")
        return " ".join([keyword for keyword in rake.get_ranked_phrases()])

    except Exception as e:
        print("Error: ", str(e))
        traceback.print_exc()


# especializado em sumarização extrema, baseado em transformer (pesado)
def theme_from_pegasus(query):
    try:
        model_name = "google/pegasus-xsum"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        tokens = tokenizer(query, truncation=True, padding="longest", return_tensors="pt")
        theme_tokens = model.generate(**tokens)
        return tokenizer.decode(theme_tokens[0], skip_special_tokens=True)

    except Exception as e:
        print("Error: ", str(e))
        traceback.print_exc()


# BERT + GPT -> faz outras coisas, porém é bom pra sumarização. Alternativa mais pesada.
def theme_from_bart(query):
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokens = tokenizer(query, truncation=True, padding="longest", return_tensors="pt")
        theme_tokens = model.generate(**tokens)
        return tokenizer.decode(theme_tokens[0], skip_special_tokens=True)

    except Exception as e:
        print("Error: ", str(e))
        traceback.print_exc()


# Todas as alternativas abaixo possuem costumização de parâmetros que podem ser úteis
SUMMARIZERS = {"yake": theme_from_yake, "rake": theme_from_rake, "pegasus": theme_from_pegasus, "bart": theme_from_bart}


# default -> theme_of_query = query
def get_theme(query, option="default"):
    if option not in SUMMARIZERS.keys():
        return query

    return SUMMARIZERS[option](query)
