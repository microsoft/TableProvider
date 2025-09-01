# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum, unique

# Tasks available in the orchestrator
@unique
class TaskName(Enum):
    feverous = "feverous"
    hybridqa = "hybridqa"
    sqa = "sqa"
    tabfact = "tabfact"
    totto = "totto"
    json_convert = "json_convert"

@unique
class TableSource(Enum):
    input = "input"
    task = "task"

# Table sampling types available in the orchestrator
@unique
class TableSamplingType(Enum):
    evenly_sample = "evenly_sample"
    clustering_sample = "clustering_sample"
    embedding_sample = "embedding_sample"
    random_sample = "random_sample"
    sequential_random_sample = "sequential_random_sample"
    table_to_text_sample = "table_to_text_sample"
    auto_row_filter = "auto_row_filter"

# Table augmentation types available in the orchestrator
@unique
class TableAugmentationType(Enum):
    table_size = "table_size"
    header_hierarchy = "header_hierarchy"
    metadata = "metadata"
    header_field_categories = "header_field_categories"
    trunk_summary = "trunk_summary"
    intermediate_NL_reasoning_steps = "intermediate_NL_reasoning_steps"
    term_explanations = "term_explanations"
    docs_references = "docs_references"
    columns_analysis = "columns_analysis"
    neural_symbolic_augmentation = "neural_symbolic_augmentation"
    retrieval_based_augmentation = "retrieval_based_augmentation"
    invalid = "None"

# Table formatting types available in the orchestrator, can be used in both composed_data and table_sampling output
@unique
class TableFormatType(Enum):
    html = "html"
    csv = "csv"
    latex = "latex"
    markdown = "markdown"
    json = "json"
    excel = "excel"
    stata = "stata"
    pickle = "pickle"
    ellipsis_txt = "ellipsis_txt"
    self_custom_json = "self_custom_json"
    invalid = "None"

# Table serialization types available in the orchestrator
@unique
class TableSerializationType(Enum):
    markdown = "markdown"
    markdown_grid = "markdown_grid"
    xml = "xml"
    html = "html"
    json = "json"
    latex = "latex"
    nl_sep = "nl_sep"