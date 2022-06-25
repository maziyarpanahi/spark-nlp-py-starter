# This is a sample Python script.
import sparknlp


def start_spark_nlp():
    spark_session = sparknlp.start()
    return spark_session


def print_versions(pyspark):
    print(f'Spark NLP is {sparknlp.version()}')
    print(f'PySpark is {pyspark.version}')


if __name__ == '__main__':
    spark = start_spark_nlp()
    print_versions(spark)

