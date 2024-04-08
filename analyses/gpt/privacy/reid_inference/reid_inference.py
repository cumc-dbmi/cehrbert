import os
from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

VOCAB_SIZE = 17870
L = 0.001
SENSITIVE_MATCH_THRESHOLD = int(VOCAB_SIZE * L)

# This number came from the tokenizer, essentially all numeric tokens that represent valid OMOP concept ids
COMMON_ATTRIBUTES = ['320128', '200219', '77670', '432867', '254761', '312437', '378253']
MAX_LEVELS = [1, 0, 0] + [1] * len(COMMON_ATTRIBUTES)
# The age groups are generated using the following bases e.g. 1 indicates using the age as is
AGE_GENERALIZATION_LEVELS = [10, 1]


def generate_lattice_bfs(top_gen_levels):  # BFS
    """
    Came from https://github.com/yy6linda/synthetic-ehr-benchmarking/blob/main/privacy_evaluation/Synthetic_risk_model_reid.py#L81
    """
    visited = [top_gen_levels]
    queue = [top_gen_levels]
    lattice = []
    while queue:
        gen_levels = queue.pop(0)
        lattice.append(gen_levels)
        for i in range(len(gen_levels)):
            if gen_levels[i] != 0:
                gen_levels_new = gen_levels.copy()
                gen_levels_new[i] -= 1
                if not gen_levels_new in visited:
                    visited.append(gen_levels_new)
                    queue.append(gen_levels_new)
    return lattice


def update_dataset(
        reid_data,
        real_sample,
        synthetic_reid_data,
        config
):
    reid_data_dup = reid_data.alias("reid_data_dup")
    real_sample_dup = real_sample.alias('real_sample_dup')
    synthetic_reid_data_dup = synthetic_reid_data.alias("synthetic_reid_data_dup")
    common_attributes_to_remove = []

    for i in range(len(config)):
        # age group
        if i == 0:
            age_level_index = config[i]
            age_group_base = AGE_GENERALIZATION_LEVELS[age_level_index]
            reid_data_dup = reid_data_dup.withColumn('age', f.ceil(f.col('age').cast('int') / age_group_base))
            synthetic_reid_data_dup = synthetic_reid_data_dup.withColumn('age', f.ceil(
                f.col('age').cast('int') / age_group_base))
            real_sample_dup = real_sample_dup.withColumn('age', f.ceil(f.col('age').cast('int') / age_group_base))
        elif i in [1, 2]:
            # gender and race are not generalized
            continue
        else:
            # this indicates that the common attribute should be removed
            if config[i] == 0:
                common_attributes_to_remove.append(COMMON_ATTRIBUTES[i - 3])

    def remove_common_attributes(concept_ids):
        comm_atts = sorted(set(concept_ids) - set(common_attributes_to_remove))
        if comm_atts:
            return '-'.join(comm_atts)
        else:
            return 'empty'

    extract_common_attributes_udf = f.udf(
        lambda concept_ids: remove_common_attributes(concept_ids), t.StringType())

    reid_data_dup = reid_data_dup.withColumn(
        'common_attributes',
        extract_common_attributes_udf('common_attributes')
    )

    real_sample_dup = real_sample_dup.withColumn(
        'common_attributes',
        extract_common_attributes_udf('common_attributes')
    )

    synthetic_reid_data_dup = synthetic_reid_data_dup.withColumn(
        'common_attributes',
        extract_common_attributes_udf('common_attributes')
    )
    return reid_data_dup, real_sample_dup, synthetic_reid_data_dup


def calculate_reid_risk_score(
        real_sample_dup,
        reid_data_dup,
        synthetic_reid_data_dup,
        lower_n,
        cap_n,
        lambda_val=0.23,
        num_salts=20
):
    real_sample_dup = real_sample_dup.withColumn(
        "salt", (f.rand() * num_salts).cast("int")
    )

    reid_data_stats = reid_data_dup \
        .groupby('age', 'gender', 'race', 'common_attributes') \
        .count()

    real_to_population_matches = real_sample_dup.join(
        reid_data_stats,
        (real_sample_dup['age'] == reid_data_stats['age']) &
        (real_sample_dup['gender'] == reid_data_stats['gender']) &
        (real_sample_dup['race'] == reid_data_stats['race']) &
        (real_sample_dup['common_attributes'] == reid_data_stats['common_attributes'])
    ).select('person_id', 'count')

    # Alias the DataFrame for self join
    real_sample_stats = real_sample_dup \
        .groupby('age', 'gender', 'race', 'common_attributes') \
        .count()

    real_to_real_matches = real_sample_dup.join(
        real_sample_stats,
        (real_sample_dup['age'] == real_sample_stats['age']) &
        (real_sample_dup['gender'] == real_sample_stats['gender']) &
        (real_sample_dup['race'] == real_sample_stats['race']) &
        (real_sample_dup['common_attributes'] == real_sample_stats[
            'common_attributes'])
    ).select('person_id', 'count')

    reid_data_step_one = reid_data_dup.join(real_to_population_matches, 'person_id') \
        .withColumnRenamed('count', 'upper_F_s') \
        .join(real_to_real_matches, 'person_id') \
        .withColumnRenamed('count', 'lower_f_s')

    synthetic_reid_data_dup = synthetic_reid_data_dup.withColumn(
        "salt", f.explode(f.array([f.lit(x) for x in range(num_salts)]))
    )

    real_sample_dup = real_sample_dup.where(f.size('sensitive_attributes') >= SENSITIVE_MATCH_THRESHOLD)
    synthetic_reid_data_dup = synthetic_reid_data_dup.where(f.size('sensitive_attributes') >= SENSITIVE_MATCH_THRESHOLD)

    real_to_synthetic = real_sample_dup.join(
        synthetic_reid_data_dup,
        (real_sample_dup['age'] == synthetic_reid_data_dup['age']) &
        (real_sample_dup['gender'] == synthetic_reid_data_dup['gender']) &
        (real_sample_dup['race'] == synthetic_reid_data_dup['race']) &
        (real_sample_dup['common_attributes'] == synthetic_reid_data_dup['common_attributes']) &
        (real_sample_dup['salt'] == synthetic_reid_data_dup['salt'])
    ).select(
        real_sample_dup['person_id'],
        real_sample_dup['sensitive_attributes'].alias('real_sensitive_attributes'),
        synthetic_reid_data_dup['sensitive_attributes'].alias('synthetic_sensitive_attributes')
    ) \
        .withColumn('n_of_sensitive_matches',
                    f.size(f.array_intersect('real_sensitive_attributes', 'synthetic_sensitive_attributes'))) \
        .drop('real_sensitive_attributes', 'synthetic_sensitive_attributes')

    real_to_synthetic = real_to_synthetic.groupby('person_id') \
        .agg(f.max('n_of_sensitive_matches').alias('n_of_sensitive_matches')) \
        .withColumn('new_info', (f.col('n_of_sensitive_matches') > f.lit(SENSITIVE_MATCH_THRESHOLD)).cast('int'))

    reid_data_step_two = reid_data_step_one.join(
        real_to_synthetic,
        'person_id',
        how='left_outer'
    ).withColumn('new_info', f.coalesce(real_to_synthetic['new_info'], f.lit(0)))

    return reid_data_step_two \
        .withColumn('A_term', 1 / f.col('lower_f_s') * f.lit((1 + lambda_val) / 2) * f.col('new_info')) \
        .withColumn('B_term', 1 / f.col('upper_f_s') * f.lit((1 + lambda_val) / 2) * f.col('new_info')) \
        .select((f.sum('A_term') / f.lit(cap_n)).alias('A_term'), (f.sum('B_term') / f.lit(lower_n)).alias('B_term'))


def main(args):
    all_configs = generate_lattice_bfs(MAX_LEVELS)

    N = None
    n = None
    excluded_sensitive_attributes = None

    for config in tqdm(all_configs, total=len(all_configs)):

        spark = SparkSession \
            .builder \
            .appName(f'Generate REID {"".join(map(str, config))}') \
            .getOrCreate()

        if sum(config) == 0:
            continue

        experiment_output_name = os.path.join(args.output_folder, f'{"".join(map(str, config))}.parquet')
        if os.path.exists(experiment_output_name):
            continue

        @f.udf(t.ArrayType(t.StringType()))
        def extract_omop_concepts_udf(concept_ids):
            return list(sorted(set([_ for _ in concept_ids if str.isnumeric(_)])))

        def extract_common_attributes(concept_ids):
            commn_atts = set([c for c in concept_ids if c in COMMON_ATTRIBUTES])
            return list(commn_atts)

        extract_common_attributes_udf = f.udf(
            lambda concept_ids: extract_common_attributes(concept_ids), t.ArrayType(t.StringType()))

        def extract_sensitive_attributes(concept_ids):
            return list(sorted(set([c for c in concept_ids if c not in COMMON_ATTRIBUTES])))

        extract_sensitive_attributes_udf = f.udf(
            lambda concept_ids: extract_sensitive_attributes(concept_ids),
            t.ArrayType(t.StringType()))

        data = spark.read.parquet(args.population_data_folder)
        data = data.select('concept_ids', 'num_of_concepts', 'person_id') \
            .withColumn('is_real', f.col('num_of_concepts') >= 20) \
            .drop('num_of_concepts')

        reid_data = data \
            .withColumn('age', f.col('concept_ids')[1]) \
            .withColumn('age', f.split('age', ':')[1]) \
            .withColumn('gender', f.col('concept_ids')[2]) \
            .withColumn('race', f.col('concept_ids')[3]) \
            .withColumn('concept_ids', extract_omop_concepts_udf('concept_ids')) \
            .withColumn('common_attributes', extract_common_attributes_udf('concept_ids')) \
            .withColumn('sensitive_attributes', extract_sensitive_attributes_udf('concept_ids')) \
            .drop('concept_ids')

        real_sample = reid_data.where('is_real')

        if not N:
            N = reid_data.count()
        if not n:
            n = real_sample.count()

        if not excluded_sensitive_attributes:
            excluded_sensitive_attributes_df = real_sample.select(
                f.explode('sensitive_attributes').alias('sensitive_attribute')
            ).groupby('sensitive_attribute').count() \
                .withColumn('sensitive_attribute_prevalence', f.col('count') / n) \
                .withColumn('is_majority', f.col('sensitive_attribute_prevalence') >= 0.5) \
                .where('is_majority')
            excluded_sensitive_attributes = [
                row.sensitive_attribute for row in
                excluded_sensitive_attributes_df.select('sensitive_attribute').collect()
            ]

        def filter_sensitive_attributes(concept_ids):
            return [c for c in concept_ids if c not in excluded_sensitive_attributes]

        filter_sensitive_attributes_udf = f.udf(
            lambda concept_ids: filter_sensitive_attributes(concept_ids),
            t.ArrayType(t.StringType()))

        real_sample = real_sample.withColumn(
            'sensitive_attributes',
            filter_sensitive_attributes_udf('sensitive_attributes')
        )

        synthetic_data = spark.read.parquet(args.synthetic_data_folder)
        # Remove [START]
        synthetic_data = synthetic_data.withColumn(
            'concept_ids', f.slice(f.col('concept_ids'), 2, 100000)
        )

        synthetic_reid_data = synthetic_data \
            .withColumn('age', f.col('concept_ids')[1]) \
            .withColumn('age', f.split('age', ':')[1]) \
            .withColumn('gender', f.col('concept_ids')[2]) \
            .withColumn('race', f.col('concept_ids')[3]) \
            .withColumn('concept_ids', extract_omop_concepts_udf('concept_ids')) \
            .withColumn('common_attributes', extract_common_attributes_udf('concept_ids')) \
            .withColumn('sensitive_attributes', extract_sensitive_attributes_udf('concept_ids')) \
            .drop('concept_ids')

        reid_data_dup, real_sample_dup, synthetic_reid_data_dup = update_dataset(
            reid_data,
            real_sample,
            synthetic_reid_data,
            config=config
        )
        experiment_risk_score = calculate_reid_risk_score(
            real_sample_dup,
            reid_data_dup,
            synthetic_reid_data_dup,
            lower_n=n,
            cap_n=N,
            lambda_val=0.23,
            num_salts=args.num_salts
        )
        experiment_risk_score.toPandas().to_parquet(experiment_output_name)

        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Arguments for re-identification risk evaluation')
    parser.add_argument(
        '--population_data_folder',
        dest='population_data_folder',
        action='store',
        required=True
    )
    parser.add_argument(
        '--synthetic_data_folder',
        dest='synthetic_data_folder',
        action='store',
        required=True
    )
    parser.add_argument(
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The output folder for storing the results',
        required=True
    )
    parser.add_argument(
        '--num_salts',
        dest='num_salts',
        action='store',
        type=int,
        default=40,
        required=False
    )
    main(parser.parse_args())
