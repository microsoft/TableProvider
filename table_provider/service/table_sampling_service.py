# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from table_provider.config.entity.azure_openai_config import AzureOpenAIConfig
from table_provider.model.table_provider_enum import TableSamplingType
from table_provider.config.entity.task_token_limit_config import TokenLimitConfig
from table_provider.model.llm.call_embedding_client import CallEmbeddingClient
from table_provider.model.llm.call_llm_client import CallLLMClient
from table_provider.utils import select_top_k_samples


def n_gram_overlap(txt_a: str, txt_b: str, n: int = 3) -> float:
    tokens_a, tokens_b = txt_a.split(), txt_b.split()
    n_grams_a = Counter(
        [' '.join(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)]
    )
    n_grams_b = Counter(
        [' '.join(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)]
    )
    intersection = sum((n_grams_a & n_grams_b).values())
    total = sum(n_grams_a.values()) + sum(n_grams_b.values())

    return intersection / total if total > 0 else 0


class TableSamplingService:
    def __init__(
        self,
        split: str,
        n_cluster: int = 5,
        top_k: int = 3,
        whether_column_grounding: bool = False,
        llm_azure_openai_config: AzureOpenAIConfig = None,
        embedding_azure_openai_config: AzureOpenAIConfig = None,
        token_limit_config: TokenLimitConfig = None
    ):
        """
        args:
            split: str, train, dev, or test
            table_sampling_type: str, row filter type
        """
        self.split = split
        self.n_cluster = n_cluster
        self.top_k = top_k
        self.whether_column_grounding = whether_column_grounding
        self.token_limit_config = TokenLimitConfig(token_limit_config)
        self.user_query = None

        self.call_llm_client = CallLLMClient(llm_azure_openai_config)
        self.call_embedding_client = CallEmbeddingClient(embedding_azure_openai_config)

    def sample(self, query: str, parsed_example: dict, table_sampling_type):
        # Check row filter type
        if table_sampling_type not in [
            sampling_type.value for sampling_type in TableSamplingType
        ] + ["default"]:
            raise ValueError(
                f"Table sampling type {table_sampling_type} is not supported"
            )
        # set the default sampling type
        if table_sampling_type == "default":
            table_sampling_type = "clustering_sample"

        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, parsed_example
        assert len(parsed_example["table"]["header"]) > 0, parsed_example
        self.user_query = query
        # Run the row filter
        return self.func_set()[table_sampling_type](parsed_example)

    def evenly_sampling(self, _example: dict) -> pd.DataFrame:
        """
        row wise insert, header, row 1, row n, row 2, row n-1, ..., row n/2, row n/2+1, until token_size_threshold is reached.
        args:
            _example: dict, parsed table
        return:
            df: pd.DataFrame, filtered table
        """
        total_token_count = 0

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm_client.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Insert rows from the head and tail of the DataFrame until the token size threshold is reached
        head_index, tail_index = 0, len(rows) - 1
        rows_count = 0
        while (
            total_token_count <= self.token_limit_config.max_truncate_tokens
            and rows_count < self.token_limit_config.max_rows
        ):
            head_row = rows[head_index]
            tail_row = rows[tail_index]
            head_token_count = self.call_llm_client.num_tokens_list(head_row)
            tail_token_count = self.call_llm_client.num_tokens_list(tail_row)

            # If the head and tail meet, add it and break the loop
            if (
                head_index >= tail_index
                or total_token_count + head_token_count
                > self.token_limit_config.max_truncate_tokens
                or total_token_count + tail_token_count
                > self.token_limit_config.max_truncate_tokens
            ):
                break

            # If adding the head row does not exceed the token size threshold, add it to the new DataFrame
            if (
                total_token_count + head_token_count
                <= self.token_limit_config.max_truncate_tokens
            ):
                df.loc[len(df.index)] = head_row
                total_token_count += head_token_count
                head_index += 1
                rows_count += 1

            # Otherwise, if adding the tail row does not exceed the token size threshold, add it to the new DataFrame
            if (
                total_token_count + tail_token_count
                <= self.token_limit_config.max_truncate_tokens
            ):
                df.loc[len(df.index)] = tail_row
                total_token_count += tail_token_count
                tail_index -= 1
                rows_count += 1

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        return df

    def clustering_sampling(
        self,
        _example: dict,
    ) -> pd.DataFrame:
        """
        Cluster rows into n clusters, and sample top k rows from each cluster.
        args:
            _example: dict, parsed table
            n_cluster: int, number of clusters
            top_k: int, number of rows to sample from each cluster
        return:
            df: pd.DataFrame, filtered table
        """
        if self.user_query == "":
            raise ValueError("User query is empty, clustering sampling is not available without user query.")

        total_token_count = 0
        n_cluster = self.n_cluster
        top_k = self.top_k

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm_client.num_tokens_list(_example["table"]["header"])

        # for all columns in the rows, turn int into str
        rows = [[str(item) if isinstance(item, (int, float)) else item for item in row] for row in rows]

        total_token_count += column_token_count

        # Generate embeddings of each rows and the user query
        rows_embeddings, user_query_embeddings = self.call_embedding_client.call_embeddings(
            user_query=self.user_query,
            row_column_list=['|'.join(row) for row in rows],
        )

        if n_cluster > len(rows):
            n_cluster = len(rows)

        # Cluster rows
        k_means = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(
            rows_embeddings
        )
        cluster_labels = k_means.labels_

        # Candidate rows from clustering closet to the user query
        candidate_rows = []
        for cluster_id in range(n_cluster):
            cluster_indices = np.where(cluster_labels == cluster_id)[
                0
            ]  # get the indices of the rows in the cluster
            rows_embeddings = np.array(rows_embeddings)  # convert to np
            valid_indices = select_top_k_samples(
                rows_embeddings[cluster_indices], user_query_embeddings, k=top_k
            )
            candidate_rows.extend(
                [np.array(rows)[cluster_indices][i] for i in valid_indices]
            )
        # Sampling based on the user query matching
        for row in candidate_rows:
            row_token_count = self.call_llm_client.num_tokens_list(row)
            if total_token_count + row_token_count <= self.token_limit_config.max_truncate_tokens:
                df.loc[len(df.index)] = row
                total_token_count += row_token_count
            else:
                break

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def embedding_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Generate embeddings of each rows and the user query, and sample rows based on the user query matching.
        args:
            _example: dict, parsed table
        return:
            df: pd.DataFrame, filtered table
        """
        if self.user_query == "":
            raise ValueError("User query is empty, embedding sampling is not available without user query.")

        total_token_count = 0

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm_client.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # for all columns in the rows, turn int into str
        rows = [[str(item) if isinstance(item, (int, float)) else item for item in row] for row in rows]

        # Generate embeddings of each rows and the user query
        rows_embeddings, user_query_embeddings = self.call_embedding_client.call_embeddings(
            user_query=self.user_query,
            row_column_list=['|'.join(row) for row in rows]
        )

        # Select the top k rows based on the user query matching
        top_k_rows = select_top_k_samples(
            rows_embeddings, user_query_embeddings, k=self.token_limit_config.max_rows
        )

        if self.whether_column_grounding:
            columns = df.columns

            # call embeddings generator
            columns_embeddings, user_query_embedding = self.call_embedding_client.call_embeddings(
                user_query=self.user_query,
                row_column_list=['|'.join(column) for column in columns],
            )

            # column candidates
            candidate_columns = [
                columns[index]
                for index in select_top_k_samples(
                    columns_embeddings,
                    user_query_embedding,
                    k=self.token_limit_config.max_columns,
                )
            ]

            # only keep the columns that are in the candidate columns
            df = df.loc[:, candidate_columns]

        # Add the top k rows to the new DataFrame
        while total_token_count <= self.token_limit_config.max_truncate_tokens:
            for top_i in top_k_rows:
                top_row = rows[top_i]
                total_token_count += self.call_llm_client.num_tokens_list(top_row)
                df.loc[len(df.index)] = top_row
            break

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def random_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Random sampling the rows.
        args:
            _example: dict, parsed table
        return:
            df: pd.DataFrame, filtered table
        """
        total_token_count = 0

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm_client.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Random sampling the rows
        visited = set()
        while total_token_count <= self.token_limit_config.max_truncate_tokens:
            random_index, random_row = random.choice(list(enumerate(rows)))
            if random_index not in visited:
                total_token_count += self.call_llm_client.num_tokens_list(random_row)
                try:
                    df.loc[len(df.index)] = random_row
                except ValueError:
                    print("ValueError: {}".format(random_row))
                    print(rows)
                    print(df)
                    break
                visited.add(random_index)
            if len(visited) == len(rows) or len(df.index) >= self.token_limit_config.max_rows:
                break

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def content_snapshot(self, _example: dict) -> pd.DataFrame:
        total_token_count = 0
        utterance = self.user_query

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm_client.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        while total_token_count <= self.token_limit_config.max_truncate_tokens:
            if self.token_limit_config.max_rows > 1:
                overlap_scores = [
                    max(n_gram_overlap(utterance, ' '.join(row)) for row in rows)
                    for row in rows
                ]
                top_k_indices = np.argsort(overlap_scores)[-self.token_limit_config.max_rows :]
                df = df.append([rows[i] for i in top_k_indices], ignore_index=True)
            else:
                # Create a synthetic row for K=1
                synthetic_row = []
                for col_index in range(len(_example["table"]["header"])):
                    column_values = [row[col_index] for row in rows]
                    overlap_scores = [
                        n_gram_overlap(utterance, value) for value in column_values
                    ]
                    best_value = column_values[np.argmax(overlap_scores)]
                    synthetic_row.append(best_value)
                df.loc[0] = synthetic_row
        return df

    def table_to_text_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Leverage GPT-3 for zero-shot table-to-text generation.
        Args:
            _example: dict, parsed table
        Return:
            df: pd.DataFrame, filtered table
        """
        df = pd.DataFrame(index=range(1), columns=_example["table"]["header"])
        df_rows = pd.DataFrame(
            _example["table"]["rows"], columns=_example["table"]["header"]
        )
        # if df's length > 3000, truncate it
        if len(df_rows) > 3000:
            df_rows = df_rows[:3000]
        for col_index, col in enumerate(df.columns):
            column = df_rows[col]
            # if column is int/float, turn it into str
            column_text = [
                str(item) if isinstance(item, (int, float)) else item for item in column
            ]
            summarized_text = self.call_llm_client.call_llm_summarization(
                "|".join(column_text)
            )
            df.iloc[0, col_index] = summarized_text

        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    @staticmethod
    def sequential_random_sampling(_example: dict):
        """
        Sequentially randomly samples n rows from the given DataFrame while maintaining the original order.

        Parameters:
        df (pd.DataFrame): The input DataFrame from which to sample rows.
        n (int): The number of rows to sample.

        Returns:
        pd.DataFrame: A new DataFrame containing the sampled rows in their original order.
        """
        table_df = pd.DataFrame(_example["table"]["rows"], columns=_example["table"]["header"])
        n = min(10, len(table_df))
        random_indices = np.random.permutation(len(table_df))
        sampled_indices = random_indices[:n]
        sampled_indices = np.sort(sampled_indices)

        sampled_df = table_df.iloc[sampled_indices].reset_index(drop=True)
        return sampled_df

    def auto_table_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Leverage GPT-3 for zero-shot row filtering program generation.
        Reference: Generate, Transform, Answer: Question Specific Tool Synthesis for Tabular Data
        Args:
            _example: dict, parsed table
        Return:
            df: pd.DataFrame, filtered table
        """
        df = pd.DataFrame(
            index=_example["table"]["rows"][:5], columns=_example["table"]["header"]
        )

        context = self.user_query + "\n\n" + df.to_string()
        code_snippet = self.call_llm_client.call_llm_code_generation(context)
        try:
            # if the code snippet is valid, then execute it
            # TODO this code
            eval(code_snippet)
            df = exec(code_snippet)
            # print("Sampled Tables:\n {}".format(df))
            return df
        except Exception as e:
            print("Error: {}".format(e))
            return df

    def func_set(self) -> dict:
        return {
            TableSamplingType.evenly_sample.value: self.evenly_sampling,
            TableSamplingType.clustering_sample.value: self.clustering_sampling,
            TableSamplingType.embedding_sample.value: self.embedding_sampling,
            TableSamplingType.random_sample.value: self.random_sampling,
            TableSamplingType.table_to_text_sample.value: self.table_to_text_sampling,
            TableSamplingType.auto_row_filter.value: self.auto_table_sampling,
            TableSamplingType.sequential_random_sample.value: self.sequential_random_sampling,
        }
