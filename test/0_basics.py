import pytest
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline

spark = sparknlp.start()

text_list = ['Peter Parker is a nice guy and lives in New York.',
             'Bruce Wayne is also a nice guy and lives in Gotham City.']

df = spark.createDataFrame(text_list, StringType()).toDF("text")

# write the target entities to txt file

entities = ['Wall Street', 'USD', 'stock', 'NYSE']
with open('./resources/financial_entities.txt', 'w') as f:
    for i in entities:
        f.write(i + '\n')

entities = ['soccer', 'world cup', 'Messi', 'FC Barcelona']
with open('./resources/sport_entities.txt', 'w') as f:
    for i in entities:
        f.write(i + '\n')

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols('document') \
    .setOutputCol('sentences')

toke = Tokenizer() \
    .setInputCols("sentences") \
    .setOutputCol("token")


@pytest.mark.fast
def test_preprocessing_pipeline():
    norm = Normalizer() \
        .setInputCols("token") \
        .setOutputCol("normalized") \
        .setLowercase(True) \
        .setCleanupPatterns(["[^\w\d\s]"]) \
        .setSlangDictionary("./resources/slangs.txt", ",")

    stopwords_cleaner = StopWordsCleaner() \
        .setInputCols("normalized") \
        .setOutputCol("cleanTokens") \
        .setStopWords(["is", "the", "a"]) \
        .setCaseSensitive(False)

    lemmatizer = Lemmatizer() \
        .setInputCols("token") \
        .setOutputCol("lemma") \
        .setDictionary("./resources/AntBNC_lemmas_ver_001.txt", value_delimiter="\t", key_delimiter="->")

    ngrams_cum = NGramGenerator() \
        .setInputCols("token") \
        .setOutputCol("ngrams") \
        .setN(3) \
        .setEnableCumulative(True) \
        .setDelimiter("_")  # Default is space

    nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        toke,
        norm,
        stopwords_cleaner,
        lemmatizer,
        ngrams_cum
    ])

    pipelineModel = nlpPipeline.fit(df)
    pipelineModel.transform(df).collect()
    assert True


@pytest.mark.fast
def test_match_pipeline():
    financial_entity_extractor = TextMatcher() \
        .setInputCols("document", 'token') \
        .setOutputCol("financial_entities") \
        .setEntities("./resources/financial_entities.txt") \
        .setCaseSensitive(False) \
        .setEntityValue('financial_entity')

    sport_entity_extractor = TextMatcher() \
        .setInputCols("document", 'token') \
        .setOutputCol("sport_entities") \
        .setEntities("./resources/sport_entities.txt") \
        .setCaseSensitive(False) \
        .setEntityValue('sport_entity')

    nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        toke,
        financial_entity_extractor,
        sport_entity_extractor
    ])

    pipelineModel = nlpPipeline.fit(df)
    pipelineModel.transform(df).collect()
    assert True
