import random
from datetime import date, timedelta


def random_slice_gpt_sequence(
        concept_ids,
        max_seq_len
):
    seq_length = len(concept_ids)
    starting_points = []
    [start_year, start_age, start_gender, start_race] = [_ for _ in concept_ids[0:4]]
    start_year = int(start_year.split(':')[1])
    start_age = int(start_age.split(':')[1])
    data_cursor = date(int(start_year), 1, 1)
    birth_date = date(start_year - start_age, 1, 1)
    for i in range(4, max(5, seq_length - max_seq_len)):
        current_token = concept_ids[i]
        if current_token == 'VS':
            starting_points.append((i, data_cursor.year, data_cursor.year - birth_date.year))
        elif current_token[0] == 'D':
            att_date_delta = int(current_token[1:])
            data_cursor = data_cursor + timedelta(days=att_date_delta)
        elif current_token == 'LT':
            att_date_delta = 365 * 3
            data_cursor = data_cursor + timedelta(days=att_date_delta)
        elif current_token[:3] == 'VS-':
            data_cursor = data_cursor + timedelta(days=int(current_token.split('-')[1][1:]))
    random_starting_index, random_starting_year, random_starting_age = random.choice(starting_points)
    demographic_tokens = [
        f'year:{random_starting_year}',
        f'age:{random_starting_age}',
        start_gender,
        start_race
    ]
    # Remove the number of demographic tokens
    random_end_index = random_starting_index
    for i in reversed(list(range(random_starting_index, random_starting_index + max_seq_len - 4))):
        current_token = concept_ids[i]
        if current_token == 'VE':
            random_end_index = i
            break
    # new_token_ids = demographic_tokens + concept_ids[random_starting_index:random_end_index + 1]

    return random_starting_index, random_end_index, demographic_tokens
