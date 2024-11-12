# SciTQA: A Challenging Question Answering Dataset on a Scientific Hybrid of Tabular and Textual Content

SciTQA (Scientific Tabular And Textual dataset for Question Answering) contains 12,790 questions associated with 12,753 hybrid contexts from arxiv papers.

You can download our SciTQA dataset via [SciTQA](./dataset).

Each question in our SciTQA dataset contains the following keys:
```python
{
        "id": The unique id of the question,
        "paragraph": {               
            # The paper paragraph related to the question (is empty if the question is only associated to the tables)
            "paragraph_id": The unique id of the paragraph,
            "text": The content of the paragraph
        },
        "tables": [                                                                                              
            # The tables related to the question (is empty if the question is only associated to text)
            {
                "table_id": The unique id of the table,
                "label": The label of the table used in the latex code of the paper,
                "caption": The caption of the table,
                "table": List[List[str]], the content of the table
            }
        ],
        "question": The question itself,
        "question_type": The question type,
        "reasoning": The reasoning rationale of the question,
        "reasoning_c": The Chinese reasoning rationale of the question (We only provide Chinses rationale in the test data. We will update Chinses rationale in train data and dev data lately.),
        "answer": The answer of the question,
        "answer_c": The Chinese answer of the question (We only provide Chinses answer in the test data. We will update Chinses answers in train data and dev data lately.)
    }

```
