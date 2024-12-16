# SciTaT: A Question Answering Benchmark for Scientific Tables and Text Covering Diverse Reasoning Types

SciTaT contains 13,808 questions associated with 8,907 arxiv papers.
You can download our SciTQA dataset via [SciTQA](./dataset).

Each question in our SciTQA dataset contains the following keys:
```python
{
        "id": The unique id of the question,
        "paragraph": {               
            # The paper paragraph related to the question
            "paragraph_id": The unique id of the paragraph,
            "text": The content of the paragraph
        },
        "tables": [                                                                                              
            # The tables related to the question
            {
                "table_id": The unique id of the table,
                "label": The label of the table used in the latex code of the paper,
                "caption": The caption of the table,
                "table": List[List[str]], the content of the table
            }
        ],
        "question": The question itself,
        "question_c": The Chinese question,
        "question_type": The question type,
        "reasoning": The reasoning rationale of the question,
        "reasoning_c": The Chinese reasoning rationale of the question,
        "answer": The answer of the question,
        "answer_c": The Chinese answer of the question
    }

```