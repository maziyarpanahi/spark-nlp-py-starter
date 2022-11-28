import pyspark
import pytest
import sparknlp
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.training import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sparknlp_version = sparknlp.version()
pyspark_version = pyspark.__version__

sparknlp_version_int = int("".join(sparknlp_version.split(".")))
pyspark_version_int = int("".join(pyspark_version.split(".")))

if sparknlp_version_int >= 400:
    spark = sparknlp.start()
else:
    if pyspark_version_int >= 320:
        spark = sparknlp.start(spark32=True)
    else:
        spark = sparknlp.start()

text_list = ['Peter Parker is a nice guy and lives in New York.',
             'Bruce Wayne is also a nice guy and lives in Gotham City.']

long_text = '''google is acquiring data science community kaggle. Sources tell us that google is acquiring kaggle, 
a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat 
vague , but given that google is hosting its Cloud Next conference in san francisco this week, the official 
announcement could come as early as tomorrow. Reached by phone, kaggle co-founder ceo anthony goldbloom declined to 
deny that the acquisition is happening. google itself declined 'to comment on rumors'. kaggle, which has about half a 
million data scientists on its platform, was founded by Goldbloom and Ben Hamner in 2010. The service got an early 
start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well 
ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science 
and machine learning competitions. With kaggle, google is buying one of the largest and most active communities for 
data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty 
of that thanks to Tensorflow and other projects). kaggle has a bit of a history with google, too, but that's pretty 
recent. Earlier this month, google and kaggle teamed up to host a $100,000 machine learning competition around 
classifying YouTube videos. That competition had some deep integrations with the google Cloud platform, 
too. Our understanding is that google will keep the service running - likely under its current name. While the 
acquisition is probably more about Kaggle's community than technology, kaggle did build some interesting tools for 
hosting its competition and 'kernels', too. On kaggle, kernels are basically the source code for analyzing data sets 
and developers can share this code on the platform (the company previously called them 'scripts'). Like similar 
competition-centric sites, kaggle also runs a job board, too. It's unclear what google will do with that part of the 
service. According to Crunchbase, kaggle raised $12.5 million (though PitchBook says it's $12.75) since its launch in 
2010. Investors in kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, google chief economist Hal 
Varian, Khosla Ventures and Yuri Milner '''

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


@pytest.mark.fast
def test_yake_extraction():
    stopwords = StopWordsCleaner().getStopWords()

    keywords = YakeKeywordExtraction() \
        .setInputCols("token") \
        .setOutputCol("keywords") \
        .setMinNGrams(1) \
        .setMaxNGrams(3) \
        .setNKeywords(20) \
        .setStopWords(stopwords)

    yake_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, toke, keywords])

    empty_df = spark.createDataFrame([['']]).toDF("text")

    yake_Model = yake_pipeline.fit(empty_df)
    light_model = LightPipeline(yake_Model)
    light_model.fullAnnotate(long_text)

    assert True


@pytest.mark.fast
def test_training_helpers():
    training_data = CoNLL().readDataset(spark, './resources/eng.testa')
    training_data.show(1)
    trainingDataSet = POS().readDataset(spark, "./resources/anc-pos-corpus-small.txt",
                                        delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                        outputTextCol="text")
    trainingDataSet.show(1)

    train_dataset = CoNLLU(lemmaCol="lemma_train").readDataset(spark, "./resources/train_small.conllu.txt")
    train_dataset.show(1)
