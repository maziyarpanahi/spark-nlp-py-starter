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

    pos_tagger = PerceptronModel.pretrained() \
        .setInputCols("document", "token") \
        .setOutputCol("pos")

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
        pos_tagger,
        ngrams_cum
    ])

    pipelineModel = nlpPipeline.fit(df)
    pipelineModel.transform(df).collect()
    assert True


@pytest.mark.slow
def test_graph_extraction():
    tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    word_embeddings = WordEmbeddingsModel.pretrained() \
        .setInputCols("document", "token") \
        .setOutputCol("embeddings")

    ner_tagger = NerDLModel.pretrained() \
        .setInputCols("document", "token", "embeddings") \
        .setOutputCol("ner")

    # MergeEntities(True) Pos, Dependency Parser and Typed Dependency Parser features under the hood
    graph_extraction = GraphExtraction() \
        .setInputCols("document", "token", "ner") \
        .setOutputCol("graph") \
        .setRelationshipTypes(["lad-PER", "lad-LOC"]) \
        .setMergeEntities(True)

    graph_pipeline = Pipeline().setStages([
        documentAssembler,
        tokenizer,
        word_embeddings,
        ner_tagger,
        graph_extraction])

    # The result dataset has a graph column with the paths between prefer,LOC relationship
    graph_data_set = graph_pipeline.fit(df).transform(df)
    graph_data_set.select("graph").collect()

    assert True


@pytest.mark.fast
def test_dep_parsers():
    tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    pos_tagger = PerceptronModel.pretrained() \
        .setInputCols("document", "token") \
        .setOutputCol("pos")

    dep_parser = DependencyParserModel.pretrained() \
        .setInputCols("document", "pos", "token") \
        .setOutputCol("dependency")

    typed_dep_parser = TypedDependencyParserModel.pretrained() \
        .setInputCols("token", "pos", "dependency") \
        .setOutputCol("dependency_type")

    dep_parser_pipeline = Pipeline(stages=[
        documentAssembler,
        tokenizer,
        pos_tagger,
        dep_parser,
        typed_dep_parser])

    pipeline_model = dep_parser_pipeline.fit(df)
    pipeline_model.transform(df).collect()
    light_model = LightPipeline(pipeline_model)
    light_model.annotate(text_list)

    assert True
