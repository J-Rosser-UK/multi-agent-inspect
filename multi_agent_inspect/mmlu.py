"""
Module: mmlu.py

This module provides functionality for evaluating multiple-choice question-answering
systems using the MMLU (Massive Multitask Language Understanding) dataset. It
defines methods to preprocess datasets, create tasks, and evaluate models or
multi-agent systems on the tasks.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.scorer import match
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset, hf_dataset
from inspect_ai._eval.eval import eval

from typing import Any, Literal, Union
from textwrap import dedent
from base import initialize_session


class EvaluateMMLU:
    """
    A class for evaluating multi-agent systems using the MMLU dataset. This class provides
    functionality to preprocess the dataset, create evaluation tasks, and compute
    metrics such as accuracy.

    Attributes
    ----------
    dataset : Dataset
        A processed dataset filtered by subject if specified.
    """

    def __init__(
        self,
        split: Union[Literal["test"], Literal["dev"], Literal["validation"]] = "test",
        shuffle: bool = False,
        subjects: Union[list[str], str] = [],
    ) -> Dataset:
        """
        Initialize the EvaluateMMLU class.

        Parameters
        ----------
        split : str, optional
            The dataset split to use. Options are "test", "dev", or "validation".
            Defaults to "test".
        shuffle : bool, optional
            Whether to shuffle the dataset. Defaults to False.
        subjects : Union[list[str], str], optional
            List of subjects to filter the dataset by. If empty, no filtering is applied.
            Defaults to [].

        Returns
        -------
        Dataset
            A processed dataset object.
        """

        dataset = hf_dataset(
            path="cais/mmlu",
            name="all",
            split=split,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=42,
        )

        # filter dataset if requested
        subjects = subjects if isinstance(subjects, list) else [subjects]
        if len(subjects) > 0:
            self.dataset = dataset.filter(
                name=f"{dataset.name}-{'-'.join(subjects)}",
                predicate=lambda sample: sample.metadata is not None
                and sample.metadata.get("subject") in subjects,
            )
        else:
            self.dataset = dataset

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing a multiple-choice question into a `Sample` object.

        Parameters
        ----------
        record : dict[str, Any]
            A dictionary containing:
            - "question": str
            - "choices": list[str]
            - "answer": int (the index of the correct choice)
            - "subject": str (the category or subject of the question)

        Returns
        -------
        Sample
            A `Sample` object with:
            - input: The constructed multiple-choice question prompt.
            - target: The correct answer letter (e.g., "A", "B", "C", etc.).
            - metadata: A dictionary containing the subject of the question.
        """

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following multiple choice question.

            {record["question"]}
        """
        ).strip()

        # Append the choices, labeling each with a letter starting at 'A'
        choices_prompt = "\n".join(
            f"({chr(65 + i)}) {choice}" for i, choice in enumerate(record["choices"])
        )

        # Combine question and choices into a single prompt
        prompt = f"{question_prompt}\n{choices_prompt}\n\n"
        prompt += f"Provide your answer as a single letter in the range A-{chr(65 + len(record['choices']) - 1)}."

        # Determine the correct answer letter
        correct_answer_letter = chr(65 + record["answer"])

        return Sample(
            input=prompt,
            target=correct_answer_letter,
            metadata={"subject": record["subject"]},
        )

    @solver
    def match_solver(self, agent_system) -> Solver:
        """
        Define a solver for the match task.

        Parameters
        ----------
        agent_system : Any
            The multi-agent system to be evaluated.

        Returns
        -------
        Solver
            A solver object that can process a TaskState using the given multi-agent system.
        """

        async def solve(state: TaskState, generate: Generate) -> TaskState:

            try:
                session, Base = initialize_session("test.db")
                system = agent_system(session)
                task = state.input
                state.output.completion = await system.forward(task)

            except Exception as e:

                print("Error during evaluation:", e)

            finally:
                session.close()

            return state

        return solve

    @task
    def match_task(self, agent_system):
        """
        Create a match task for evaluation.

        Parameters
        ----------
        agent_system : Any
            The multi-agent system to be evaluated.

        Returns
        -------
        Task
            A task object that combines the dataset, solver, and scorer.
        """
        return Task(
            dataset=self.dataset,
            solver=self.match_solver(agent_system),
            scorer=match(),
            config=GenerateConfig(temperature=0.5),
        )

    def evaluate(self, agent_system, limit=100):
        """
        Evaluate the given multi-agent system on the dataset.

        Parameters
        ----------
        agent_system : Any
            The multi-agent system to be evaluated.
        limit : int, optional
            The maximum number of samples to evaluate. Defaults to 100.

        Returns
        -------
        float
            The accuracy score of the multi-agent system on the dataset.
        """

        results = eval(
            self.match_task(agent_system),
            model="openai/gpt-3.5-turbo",  # this doesn't matter and isn't used
            limit=limit,
            log_dir="./logs",  # specify where logs are stored
            log_format="eval",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
        )

        # 'results' is a list of EvalLog objects (usually one per task)
        # Each EvalLog contains metrics for the entire task/dataset.
        accuracy = -2

        for res in results:
            if res.results and res.results.scores:
                print("Final metrics for the entire dataset:")
                for score in res.results.scores:
                    print(f"Score: {score.name}")
                    for metric_name, metric in score.metrics.items():
                        print(f"  {metric_name}: {metric.value}")
                        if metric_name == "accuracy":
                            accuracy = metric.value

        return accuracy

    def evaluate_multiple(self, agent_systems_list, limit=100):
        """
        Evaluate multiple multi-agent systems on the dataset.

        Parameters
        ----------
        agent_systems_list : list
            A list of multi-agent systems to evaluate.
        limit : int, optional
            The maximum number of samples to evaluate. Defaults to 100.

        Returns
        -------
        None
            Prints evaluation results for each multi-agent system.
        """

        tasks = []
        for system in agent_systems_list:
            tasks.append(self.match_task(system))

        results = eval(
            tasks,
            model="openai/gpt-3.5-turbo",  # this doesn't matter and isn't used
            limit=limit,
            log_dir="./logs",  # specify where logs are stored
            log_format="eval",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
        )

        print(results)
