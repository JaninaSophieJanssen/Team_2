import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = repo_root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    # Fix Google Cloud credentials path
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        cred_file = Path(cred_path)
        # Check if file exists at the given path
        if not cred_file.exists():
            # Try to find it in repo root
            filename = cred_file.name
            repo_cred_path = repo_root / filename
            if repo_cred_path.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(repo_cred_path)
                print(f"Fixed credentials path to: {repo_cred_path}")
            else:
                print(f"Warning: Credentials file not found at {cred_path} or {repo_cred_path}")
        else:
            print(f"Using credentials at: {cred_file}")
    else:
        # No env var set, look for credentials file in repo root
        default_cred = repo_root / "inner-radius-483716-p8-1e0e5d8295cc.json"
        if default_cred.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_cred)
            print(f"Set credentials path to: {default_cred}")
    return (repo_root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TSAR 2025 LLM Baseline

    This notebook generates text simplifications for the TSAR 2025 shared task using **Gemini via Vertex AI**.

    **Task:** Simplify a given paragraph to a target CEFR level (A2 or B1) while preserving the original meaning.

    **Evaluation criteria:**
    - CEFR Compliance (classifier checks if output matches target level)
    - Meaning Preservation (semantic similarity to source)
    - Similarity to References (semantic similarity to gold references)
    """)
    return


@app.cell
def _(repo_root):
    import pandas as pd

    trial_path = repo_root / "data" / "tsar2025_trialdata.jsonl"
    test_path = repo_root / "data" / "tsar2025_test.jsonl"

    trial_df = pd.read_json(trial_path, lines=True)
    trial_data = trial_df.to_dict("records")

    if test_path.exists():
        test_df = pd.read_json(test_path, lines=True)
        test_data = test_df.to_dict("records")
    else:
        test_data = []

    print(f"Loaded {len(trial_data)} trial examples, {len(test_data)} test examples")
    return test_data, test_path, trial_data, trial_path


@app.cell
def _(trial_data):
    trial_data[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM Configuration

    Select the Gemini model and generation parameters.
    """)
    return


@app.cell
def _(mo):
    gemini_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    model_selector = mo.ui.dropdown(
        options=gemini_models, value="gemini-2.5-flash", label="Select Model"
    )

    temperature_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.0,
        label="Temperature",
        show_value=True,
    )

    mo.vstack([model_selector, temperature_slider])
    return model_selector, temperature_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prompt Template

    ### What could be improved:
    1. Evaluation criterion 3 can confuse the model in a zero-shot setup, because without an example it may mix up the original_text with what should have been an example.

    2. Using XML tags to mark example boundaries would likely work better than plain text. Also, indexing examples doesn’t make much sense when only a single example is provided.
    """)
    return


@app.cell
def _(trial_data):
    from pydantic import BaseModel

    class SimplificationResponse(BaseModel):
        simplified_a2: str
        simplified_b1: str

    SYSTEM_PROMPT = """You are an expert text simplification assistant. Your task is to rewrite complex paragraphs into simpler versions that match a specific CEFR reading level while preserving the original meaning.

    CEFR Levels:
    - A2 (Elementary): Use very simple words and short sentences. Avoid complex grammar. Define or explain any necessary technical terms.
    - B1 (Intermediate): Use relatively simple vocabulary and sentence structures. Some complex sentences are acceptable but keep them clear.

    Evaluation Criteria (optimize for all three):
    1. CEFR Compliance: Your output MUST match the target CEFR level (A2 or B1) - this will be verified by a classifier
    2. Meaning Preservation: Preserve ALL key information and meaning from the original text - semantic similarity will be measured
    3. Style Consistency: Follow the simplification style shown in the examples - your output should be similar to reference simplifications

    Guidelines:
    - Study the examples carefully to understand the appropriate simplification style for each CEFR level
    - For A2: Use very basic vocabulary, short sentences, explain difficult concepts
    - For B1: Use clear language, moderate sentence complexity, maintain flow
    - Keep all important facts, numbers, and key details from the original"""

    def build_prompt(original_text: str, num_examples: int = 2) -> str:
        # Build paired examples from trial data (A2 + B1 per original)
        by_original = {}
        for item in trial_data:
            orig = item["original"]
            entry = by_original.setdefault(orig, {})
            entry[item["target_cefr"].lower()] = item["reference"]

        pairs = []
        for orig, refs in by_original.items():
            if "a2" in refs and "b1" in refs:
                pairs.append({"original": orig, "a2": refs["a2"], "b1": refs["b1"]})

        examples = ""
        if num_examples > 0 and pairs:
            selected_pairs = pairs[:num_examples]
            blocks = []
            for idx, ex in enumerate(selected_pairs, start=1):
                blocks.append(
                    f"""<example_{idx}>
    <original_text>
    {ex['original']}
    </original_text>
    <simplified_a2>
    {ex['a2']}
    </simplified_a2>
    <simplified_b1>
    {ex['b1']}
    </simplified_b1>
    </example_{idx}>"""
                )

            examples = (
                "Here are examples of text simplification for the same original text "
                "at different CEFR levels:\n\n"
                + "\n\n".join(blocks)
                + "\n\n---"
            )

        return f"""{examples}

    Now simplify the following paragraph to BOTH CEFR levels (A2 and B1).
    Your response must be a single JSON object with the keys "simplified_a2" and "simplified_b1".

    <original_text>
    {original_text}
    </original_text>"""
    return SYSTEM_PROMPT, SimplificationResponse, build_prompt


@app.cell
def _(repo_root):
    import json
    import time
    from litellm import completion
    import instructor

    instructor_client = instructor.from_litellm(completion)

    submissions_dir = repo_root / "evaluation" / "submissions" / "Team_2"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    return instructor_client, json, submissions_dir, time


@app.cell
def _(instructor_client, json, repo_root, submissions_dir, time):
    def generate_predictions(
        data,
        model_name,
        num_examples,
        temperature,
        system_prompt,
        build_prompt_fn,
        response_format,
        output_filename,
        status_obj=None,
    ):
        """Generate simplifications for a dataset and save to submission folder."""
        predictions = []

        # Group by original text to avoid duplicate API calls
        # Since we generate both A2 and B1 in one call, we only need to process unique originals
        original_to_items = {}
        for item in data:
            original_text = item["original"]
            if original_text not in original_to_items:
                original_to_items[original_text] = []
            original_to_items[original_text].append(item)

        unique_originals = list(original_to_items.keys())

        for i, original_text in enumerate(unique_originals):
            items_for_this_original = original_to_items[original_text]

            # Update status if provided
            if status_obj:
                text_ids = [item['text_id'] for item in items_for_this_original]
                status_obj.update(
                    title=f"Processing {i+1}/{len(unique_originals)}: {', '.join(text_ids)}"
                )

            # Build prompt (same for all items with this original)
            user_prompt = build_prompt_fn(original_text, num_examples)
            # Call LLM once for this original text
            try:
                # Use instructor for reliable structured output
                simplified_data = instructor_client.chat.completions.create(
                    model=f"vertex_ai/{model_name}",
                    response_model=response_format,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    vertex_project="inner-radius-483716-p8",
                    vertex_location="us-central1",
                    temperature=temperature,
                    max_tokens=2048,
                    max_retries=5,
                )

                # Create predictions for all items that share this original
                for item in items_for_this_original:
                    target_level = item["target_cefr"].lower()
                    # instructor returns a Pydantic model
                    simplified = getattr(simplified_data, f'simplified_{target_level}')

                    predictions.append(
                        {
                            "text_id": item["text_id"],
                            "simplified": simplified,
                        }
                    )

            except Exception as e:
                error = f"[ERROR: {e}]"
                print(error)


            time.sleep(0.1)  # Rate limiting

        # Save to submission folder
        output_path = submissions_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        print(f"Saved {len(predictions)} predictions to {output_path.relative_to(repo_root)}")
        return predictions
    return (generate_predictions,)


@app.cell
def _():
    SHOT_LABELS = [(0, "zero"), (1, "one"), (2, "two")]
    return (SHOT_LABELS,)


@app.cell
def _(
    SYSTEM_PROMPT,
    SimplificationResponse,
    build_prompt,
    instructor_client,
    mo,
    model_selector,
    temperature_slider,
    trial_data,
):


    # Select first item for testing
    trial_orig = trial_data[5]["original"]
    examples = []

    test_prompt = build_prompt(trial_orig, num_examples=0)
    examples.append(mo.ui.text_area(value=test_prompt, label="Example Prompt", full_width=True)) 
    _res = instructor_client.chat.completions.create(
        model=f"vertex_ai/{model_selector.value}",
        response_model=SimplificationResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt},
        ],
        vertex_project="inner-radius-483716-p8",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=2048,
        max_retries=3,
    )
    examples.append(mo.ui.text_area(value=str(_res), label="Example Answer", full_width=True))
    mo.vstack(examples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Simplifications (Trial Data)

    Run the LLM on trial data to iterate on the prompt.
    """)
    return


@app.cell
def _(mo):
    run_trial_multishot_button = mo.ui.run_button(
        label="Generate Trial Predictions (0/1/2-shot)"
    )
    run_trial_multishot_button
    return (run_trial_multishot_button,)


@app.cell
def _(
    SHOT_LABELS,
    SYSTEM_PROMPT,
    SimplificationResponse,
    build_prompt,
    generate_predictions,
    mo,
    model_selector,
    run_trial_multishot_button,
    temperature_slider,
    trial_data,
):
    trial_predictions_by_shot = {}

    if run_trial_multishot_button.value:
        for trial_num_examples, trial_shot_label in SHOT_LABELS:
            with mo.status.spinner(
                title=f"Generating trial {trial_shot_label}-shot predictions..."
            ) as trial_generation_status:
                trial_predictions_by_shot[trial_shot_label] = generate_predictions(
                    data=trial_data,
                    model_name=model_selector.value,
                    num_examples=trial_num_examples,
                    temperature=temperature_slider.value,
                    system_prompt=SYSTEM_PROMPT,
                    build_prompt_fn=build_prompt,
                    response_format=SimplificationResponse,
                    output_filename=f"trial_data_{trial_shot_label}_shot.jsonl",
                    status_obj=trial_generation_status,
                )
    else:
        print("Click 'Generate Trial Predictions (0/1/2-shot)' to run")
    return


@app.cell
def _(mo, repo_root, submissions_dir):
    import subprocess
    import sys

    def run_evaluation(gold_file_path, output_filename, submission_filename, button_value):
        """
        Shared function to run evaluation script.

        Args:
            gold_file_path: Path to gold standard file
            output_filename: Name of output Excel file (e.g., "results_trial.xlsx")
            submission_filename: Name of submission file to evaluate (e.g., "dev_trial.jsonl")
            button_value: Button value to check if evaluation should run

        Returns:
            marimo markdown display object with results
        """
        if button_value and (submissions_dir / submission_filename).exists():
            eval_script = repo_root / "evaluation" / "tsar2025_evaluation_script.py"

            env = {
                **dict(__import__("os").environ),
                "TSAR_GOLD_FILE": str(gold_file_path),
                "TSAR_SUBMISSIONS_DIR": str(submissions_dir.parent),
                "TSAR_SUBMISSION_FILE": submission_filename,  # Only evaluate this specific file
                "TSAR_OUTPUT_FILE": output_filename,
            }

            print(f"Running evaluation script: {eval_script}")
            print(f"Gold file: {gold_file_path}")
            print(f"Submissions dir: {submissions_dir.parent}")
            print(f"Output file: {output_filename}")

            with mo.status.spinner(title="Running evaluation (this may take a while)..."):
                result = subprocess.run(
                    [sys.executable, str(eval_script)],
                    capture_output=True,
                    text=True,
                    cwd=str(repo_root / "evaluation"),
                    env=env,
                )
                eval_output = result.stdout + "\n" + result.stderr

            print(f"Return code: {result.returncode}")
            print(f"Output length: {len(eval_output)}")

            if eval_output.strip():
                return mo.md(f"```\n{eval_output}\n```")
            else:
                return mo.md("*Evaluation completed but produced no output*")
        elif button_value:
            return mo.md(f"*No submission file found: {submission_filename}. Generate predictions first.*")
        else:
            return mo.md("*Click the run button to start evaluation.*")
    return (run_evaluation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Evaluation

    Execute the evaluation script on trial data.
    """)
    return


@app.cell
def _(mo):
    run_eval_multishot_button = mo.ui.run_button(
        label="Run Evaluation (0/1/2-shot)"
    )
    run_eval_multishot_button
    return (run_eval_multishot_button,)


@app.cell
def _(SHOT_LABELS, mo, run_eval_multishot_button, run_evaluation, trial_path):
    if run_eval_multishot_button.value:
        trial_eval_outputs = []
        for _, trial_eval_shot_label in SHOT_LABELS:
            trial_eval_outputs.append(mo.md(f"### Trial {trial_eval_shot_label}-shot"))
            trial_eval_outputs.append(
                run_evaluation(
                    gold_file_path=trial_path,
                    output_filename=f"results_trial_{trial_eval_shot_label}_shot.xlsx",
                    submission_filename=f"trial_data_{trial_eval_shot_label}_shot.jsonl",
                    button_value=True,
                )
            )
        trial_eval_display = mo.vstack(trial_eval_outputs)
    else:
        trial_eval_display = mo.md("*Click the run button to start evaluation.*")

    trial_eval_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Final Test Submission

    Generate predictions on the full test set.
    """)
    return


@app.cell
def _(mo):
    run_test_multishot_button = mo.ui.run_button(
        label="Generate Test Predictions (0/1/2-shot)"
    )
    run_test_multishot_button
    return (run_test_multishot_button,)


@app.cell
def _(
    SHOT_LABELS,
    SYSTEM_PROMPT,
    SimplificationResponse,
    build_prompt,
    generate_predictions,
    mo,
    model_selector,
    run_test_multishot_button,
    temperature_slider,
    test_data,
):
    test_predictions_by_shot = {}

    if run_test_multishot_button.value and test_data:
        for test_num_examples, test_shot_label in SHOT_LABELS:
            print(f'Progres:{test_num_examples}')
            if test_num_examples in [0,2]:
                continue
            with mo.status.spinner(
                title=f"Generating test {test_shot_label}-shot predictions..."
            ) as test_generation_status:
                test_predictions_by_shot[test_shot_label] = generate_predictions(
                    data=test_data,
                    model_name=model_selector.value,
                    num_examples=test_num_examples,
                    temperature=temperature_slider.value,
                    system_prompt=SYSTEM_PROMPT,
                    build_prompt_fn=build_prompt,
                    response_format=SimplificationResponse,
                    output_filename=f"test_data_{test_shot_label}_shot.jsonl",
                    status_obj=test_generation_status,
                )
    elif run_test_multishot_button.value:
        print("No test data available")
    else:
        print("Click 'Generate Test Predictions (0/1/2-shot)' to run on test set")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Test Evaluation

    Execute the evaluation script on test data.
    """)
    return


@app.cell
def _(mo):
    run_test_eval_multishot_button = mo.ui.run_button(
        label="Run Test Evaluation (0/1/2-shot)"
    )
    run_test_eval_multishot_button
    return (run_test_eval_multishot_button,)


@app.cell
def _(
    SHOT_LABELS,
    mo,
    run_evaluation,
    run_test_eval_multishot_button,
    test_path,
):
    if run_test_eval_multishot_button.value:
        test_eval_outputs = []
        for _, test_eval_shot_label in SHOT_LABELS:
            test_eval_outputs.append(mo.md(f"### Test {test_eval_shot_label}-shot"))
            test_eval_outputs.append(
                run_evaluation(
                    gold_file_path=test_path,
                    output_filename=f"results_test_{test_eval_shot_label}_shot.xlsx",
                    submission_filename=f"test_data_{test_eval_shot_label}_shot.jsonl",
                    button_value=True,
                )
            )
        test_eval_display = mo.vstack(test_eval_outputs)
    else:
        test_eval_display = mo.md("*Click the run button to start evaluation.*")

    test_eval_display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Failed Samples
    """)
    return


@app.cell
def _(json, mo, submissions_dir, test_data):
    # Get all expected IDs from the test set
    expected_ids = {item["text_id"] for item in test_data}
    missing_items = []

    for file_path in submissions_dir.glob("test_data_*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            actual_ids = {json.loads(line)["text_id"] for line in f if line.strip()}

        missing = expected_ids - actual_ids
        if missing:
            print(f"File '{file_path.name}' is missing {len(missing)} IDs.")
            original_texts = {
                a["original"] for a in test_data if a["text_id"] in missing
            }
            for text in original_texts:
                missing_items.append(
                    mo.ui.text_area(value=text, label="Original Text", full_width=True)
                )

    mo.vstack(missing_items) if missing_items else None
    return


if __name__ == "__main__":
    app.run()
