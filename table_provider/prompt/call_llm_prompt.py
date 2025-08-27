# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class CallLLMPromptType:
    """Enum class to manage all prompt types for LLM calls."""
    PARSE_TEXT_INTO_TABLE = "PARSE_TEXT_INTO_TABLE"
    FILL_IN_CELL_WITH_CONTEXT = "FILL_IN_CELL_WITH_CONTEXT"
    FILL_IN_COLUMN_WITH_CONTEXT = "FILL_IN_COLUMN_WITH_CONTEXT"
    CALL_LLM_SUMMARIZATION = "CALL_LLM_SUMMARIZATION"
    CALL_LLM_CODE_GENERATION = "CALL_LLM_CODE_GENERATION"


class CallLLMPrompt:
    """Class to manage all prompts for LLM calls."""

    PARSE_TEXT_INTO_TABLE = (
        "Example: A table summarizing the fruits from Goocrux:\n\n"
        "There are many fruits that were found on the recently discovered planet Goocrux. "
        "There are neoskizzles that grow there, which are purple and taste like candy. "
        "There are also loheckles, which are a grayish blue fruit and are very tart, "
        "a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. "
        "There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. "
        "Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, "
        "and a pale orange tinge to them.\n\n| Fruit | Color | Flavor | \n\n {string} \n\n"
    )

    FILL_IN_CELL_WITH_CONTEXT = (
        "Example: Fill in the blank based on the column context. "
        "\n\n remaining | 4 | 32 | 59 | 8 | 113 | none | 2 | 156. \n\n 16 \n\n "
        "Fill in the blank based on the column context {context}, Only return the value: "
    )

    FILL_IN_COLUMN_WITH_CONTEXT = (
        "Example: Fill in the blank (column name) based on the cells context. "
        "\n\n none | 4 | 32 | 59 | 8 | 113 | 3 | 2 | 156. \n\n remarking \n\n "
        "Fill in the blank based on the column cells context {context}, Only return the column name: "
    )

    CALL_LLM_SUMMARIZATION = (
        "Example: Summarize the table context to a single sentence. "
        "{context} \n\n: "
    )

    CALL_LLM_CODE_GENERATION = (
        "Example: Synthesize code snippet from the table context to select the proper rows. \n "
        "User 1: \nI need an expert to help me answer the question by making the table smaller. Question: Who are all of the players on the Westchester High School club team? \n "
        "table = 'Player': ['Jarrett Jack', 'Jermaine Jackson', ...\n"
        "'No.': ['1', '8', ..., 'Nationality': ['United States', 'United States', ...\n"
        "'Position': ['Guard', 'Guard', ... \n"
        "'Years in Toronto': ['2009-10', '2002-03', ... \n"
        "'School/Club Team': ['Georgia Tech', 'Detroit', ... \n"
        "User 2: \n"
        "For 'Who are all of the players on the Westchester High School club team?' the most impactful change will be to filter the rows. Since I "
        "don't know all the rows I'll use rough string matching, float casting, lowering and be as broad as possible.\n"
        ">>> new_table = table[table.apply(lambda row_dict: 'Westchester' in "
        "row_dict['School/Club Team'].lower(), axis=1)] \n "
        "Synthesize code snippet from the table context to select the proper rows\n Only return the code \n {context} \n\n: "
    )

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """Get a prompt by name, with optional parameters for formatting."""
        prompt = getattr(cls, prompt_name, None)
        if prompt:
            return prompt.format(**kwargs)
        raise ValueError(f"Prompt '{prompt_name}' not found.")