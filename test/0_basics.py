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


@pytest.mark.fast
def test_preprocessing_pipeline():
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentenceDetector = SentenceDetector() \
        .setInputCols('document') \
        .setOutputCol('sentences')

    toke = Tokenizer() \
        .setInputCols("sentences") \
        .setOutputCol("token")

    norm = Normalizer() \
        .setInputCols("token") \
        .setOutputCol("normalized") \
        .setLowercase(True) \
        .setCleanupPatterns(["[^\w\d\s]"]) \
        .setSlangDictionary("./resources/slangs.txt", ",")

    nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        toke,
        norm
    ])

    pipelineModel = nlpPipeline.fit(df)
    pipelineModel.transform(df).collect()
    assert True
