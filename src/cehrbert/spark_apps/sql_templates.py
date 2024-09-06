measurement_unit_stats_query = """
WITH measurement_percentile AS
(
    SELECT
        m.measurement_concept_id,
        m.unit_concept_id,
        MEAN(m.value_as_number) AS mean_value,
        MIN(m.value_as_number) AS min_value,
        MAX(m.value_as_number) AS max_value,
        percentile_approx(m.value_as_number, 0.01) AS lower_bound,
        percentile_approx(m.value_as_number, 0.99) AS upper_bound
    FROM measurement AS m
    WHERE EXISTS (
        SELECT
            1
        FROM required_measurement AS r
        WHERE r.measurement_concept_id = m.measurement_concept_id
            AND r.is_numeric = true
    )
    GROUP BY m.measurement_concept_id, m.unit_concept_id
)

SELECT
    m.measurement_concept_id,
    m.unit_concept_id,
    MEAN(m.value_as_number) AS value_mean,
    STDDEV(m.value_as_number) AS value_stddev,
    COUNT(*) AS measurement_freq,
    FIRST(mp.lower_bound) AS lower_bound,
    FIRST(mp.upper_bound) AS upper_bound
FROM measurement AS m
JOIN measurement_percentile AS mp
    ON m.measurement_concept_id = mp.measurement_concept_id
        AND m.unit_concept_id = mp.unit_concept_id
WHERE
    m.value_as_number BETWEEN mp.lower_bound AND mp.upper_bound
    AND m.visit_occurrence_id IS NOT NULL
    AND m.unit_concept_id <> 0
    AND m.measurement_concept_id <> 0
GROUP BY m.measurement_concept_id, m.unit_concept_id
HAVING COUNT(*) >= 100
"""
