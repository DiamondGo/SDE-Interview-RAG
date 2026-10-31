from sde_rag.parse.clean import clean_markdown
from sde_rag.parse.markdown_qa import is_question, parse_markdown_qa
from sde_rag.taxonomy import rule_tags, taxonomy
from sde_rag.text import clean_question, question_hash


def test_heading_questions_with_subheadings_in_answer():
    md = """# 100 Python Questions
#### You can also find all answers here
## 1. What are the _key features_ of Python?
Python is **interpreted**.
### Key Features
#### 1. Interpreted
Runs line by line.
## 2. How is Python executed?
Bytecode on a VM.
"""
    qas = parse_markdown_qa(md)
    assert [q.question for q in qas] == ["What are the key features of Python?", "How is Python executed?"]
    assert "Interpreted" in qas[0].answer and "Runs line by line" in qas[0].answer
    assert qas[0].section_path == "100 Python Questions"


def test_numbered_list_headings_with_indented_answers_and_unclosed_fence():
    md = """# JS Questions
1. ### What is a closure

    A closure is a function with its scope.

    ```js
    function f() {}
    ```
    stray text
    ```

2. ### What is the freeze method

    It freezes an object.
"""
    qas = parse_markdown_qa(md)
    assert [q.question for q in qas] == ["What is a closure", "What is the freeze method"]
    assert qas[0].answer.startswith("A closure")       # dedented
    assert qas[1].answer == "It freezes an object."


def test_details_summary_blocks():
    md = """## Linux
<details>
<summary>What is Linux?</summary><br><b>
A family of operating systems.
</b></details>

<details>
<summary>Explain what each of the following commands does:
  * ls
</summary><br><b>
ls lists files.
</b></details>
"""
    qas = parse_markdown_qa(md)
    assert qas[0].question == "What is Linux?" and qas[0].answer == "A family of operating systems."
    assert qas[1].question.startswith("Explain what each of the following commands does")
    assert qas[0].section_path == "Linux"


def test_bare_list_questions_and_toc_links():
    md = """* What is CSS selector specificity and how does it work?
* [What is a closure?](#what-is-a-closure)
* Describe z-index and how stacking context is formed.
* Flexbox
"""
    qas = parse_markdown_qa(md)
    assert [q.question for q in qas] == ["What is CSS selector specificity and how does it work?",
                                        "Describe z-index and how stacking context is formed."]
    assert parse_markdown_qa(md, bare_list_questions=False) == []


def test_code_comments_are_not_headings():
    md = """## How do you build the app?

    # Build, test and run
    $ npm run build

```bash
# What does this print?
echo hi
```
"""
    qas = parse_markdown_qa(md)
    assert [q.question for q in qas] == ["How do you build the app?"]
    assert "npm run build" in qas[0].answer and "echo hi" in qas[0].answer


def test_post_titles_are_not_questions_in_experience_mode():
    md = "# Ask HN: How do you interview devs now?\n\nSome text\n\n## Q1: Implement an LRU cache?\n\nUse a map."
    assert [q.question for q in parse_markdown_qa(md, title_questions=False)] == ["Implement an LRU cache?"]


def test_is_question():
    assert is_question("What is the difference between let and var")
    assert is_question("Design Pastebin.com (or Bit.ly)")
    assert is_question("Implement a rate limiter")
    assert not is_question("Build, test and run")
    assert not is_question("State management")
    assert not is_question("How to contribute")
    assert not is_question("Design Patterns")


def test_clean_question_and_hash():
    assert clean_question("📌 Q1: Why do transformers use positional embeddings?") == \
        "Why do transformers use positional embeddings?"
    assert clean_question("Q. What is Deadlock?") == "What is Deadlock?"
    assert clean_question("[E] What's the dot product?") == "What's the dot product?"
    assert clean_question("12. What is 1.5 times 2?") == "What is 1.5 times 2?"
    assert question_hash("What is a **closure**?") == question_hash("what is a closure")
    assert clean_question("Explain _concurrency_ in *Go*, see snake_case_name") == \
        "Explain concurrency in Go, see snake_case_name"


def test_clean_markdown_removes_noise_but_keeps_details():
    md = "---\ntitle: x\n---\n<!-- hi -->\n[![b](i)](l) ![img](x.png)\n**[⬆ Back to Top](#toc)**\n<details><summary>Q?</summary>A</details>\n"
    out = clean_markdown(md)
    assert "title" not in out and "hi" not in out and "Back to Top" not in out and "img" not in out
    assert "<details><summary>Q?</summary>A</details>" in out


def test_rule_tags_avoid_ordinary_words():
    assert "golang" not in rule_tags("You need to go about upgrading ntpd; should it go on the server?")
    assert "golang" in rule_tags("How do goroutines communicate?")
    assert "rate-limiter" in rule_tags("Design a rate limiter for an API")
    assert "aws" not in rule_tags("What is a lambda function in Python?")
    tax = taxonomy()
    assert tax.resolve("go") == "golang" and tax.resolve("K8s") == "kubernetes" and tax.resolve("nope") is None
