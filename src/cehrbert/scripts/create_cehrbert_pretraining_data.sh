#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER --start_date START_DATE"
    echo ""
    echo "Required Arguments:"
    echo "  --input_folder PATH      Input folder path"
    echo "  --output_folder PATH     Output folder path"
    echo "  --start_date DATE        Start date"
    echo ""
    echo "Example:"
    echo "  $0 --input_folder /path/to/input --output_folder /path/to/output --start_date 1985-01-01"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize variables
INPUT_FOLDER=""
OUTPUT_FOLDER=""
START_DATE=""

# Domain tables (fixed list)
DOMAIN_TABLES=("condition_occurrence" "procedure_occurrence" "drug_exposure")

# Parse command line arguments
ARGS=$(getopt -o "" --long input_folder:,output_folder:,start_date:,help -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$ARGS"

while true; do
    case "$1" in
        --input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --start_date)
            START_DATE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ] || [ -z "$START_DATE" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Step 1: Generate included concept list
CONCEPT_LIST_CMD="python -u -m cehrbert_data.apps.generate_included_concept_list \
-i \"$INPUT_FOLDER\" \
-o \"$OUTPUT_FOLDER\" \
--min_num_of_patients 100 \
--ehr_table_list ${DOMAIN_TABLES[@]}"

echo "Running concept list generation:"
echo "$CONCEPT_LIST_CMD"
eval "$CONCEPT_LIST_CMD"

# Step 2: Generate training data
TRAINING_DATA_CMD="python -m cehrbert_data.apps.generate_training_data \
--input_folder \"$INPUT_FOLDER\" \
--output_folder \"$OUTPUT_FOLDER\" \
-d $START_DATE \
--att_type day \
--inpatient_att_type day \
-iv \
-ip \
--include_concept_list \
--include_death \
--gpt_patient_sequence \
--domain_table_list ${DOMAIN_TABLES[@]}"

echo "Running training data generation:"
echo "$TRAINING_DATA_CMD"
eval "$TRAINING_DATA_CMD"
