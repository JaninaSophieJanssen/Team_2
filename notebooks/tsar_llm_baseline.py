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
    return test_data, trial_data, trial_path


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

    A strong baseline prompt using clear instructions, XML separation, and explicit CEFR-level targeting.
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

    def build_prompt(original_text: str) -> str:
        # Load few-shot examples from trial data (data is already uppercased in loading cell)
        a2_example = next((item for item in trial_data if item["target_cefr"] == "a2"), None)
        b1_example = next((item for item in trial_data if item["target_cefr"] == "b1"), None)

        if a2_example and b1_example:
            examples = f"""Here are examples of text simplification for the same original text at different CEFR levels:

    <example_1_a2>
    <original>
    {a2_example['original']}
    </original>
    <simplified>
    {a2_example['reference']}
    </simplified>
    </example_1_a2>

    <example_2_b1>
    <original>
    {b1_example['original']}
    </original>
    <simplified>
    {b1_example['reference']}
    </simplified>
    </example_2_b1>

    ---"""
        else:
            examples = ""

        return f"""{examples}

    Now simplify the following paragraph to BOTH CEFR levels (A2 and B1).
    This helps you understand the differentiation between levels better.

    <original_text>
    {original_text}
    </original_text>

    Provide both:
    - simplified_a2: Simplified version at A2 level
    - simplified_b1: Simplified version at B1 level"""
    return SYSTEM_PROMPT, SimplificationResponse, build_prompt


@app.cell
def _(repo_root):
    import json
    import time
    from litellm import completion

    submissions_dir = repo_root / "evaluation" / "submissions" / "Team_2"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    return completion, json, submissions_dir, time


@app.cell
def _(completion, json, repo_root, submissions_dir, time):
    def generate_predictions(
        data,
        model_name,
        temperature,
        system_prompt,
        build_prompt_fn,
        response_format,
        output_filename,
        status_obj=None,
    ):
        """Generate simplifications for a dataset and save to submission folder."""
        predictions = []

        for i, item in enumerate(data):
            # Update status if provided
            if status_obj:
                status_obj.update(
                    title=f"Processing {i+1}/{len(data)}: {item['text_id']}"
                )

            # Build prompt
            user_prompt = build_prompt_fn(item["original"])

            # Call LLM (fixed vertex project/location)
            try:
                response = completion(
                    model=f"vertex_ai/{model_name}",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    vertex_project="inner-radius-483716-p8",
                    vertex_location="us-central1",
                    temperature=temperature,
                    max_tokens=2048,
                    response_format=response_format,
                )

                # Extract simplified text based on target level
                string_repsonse = response.choices[0].message.content

                # Convert to dict (assume Pydantic v2)
                simplified_data = json.loads(string_repsonse)

                # Extract the correct level
                target_level = item["target_cefr"].lower()
                simplified = simplified_data[f'simplified_{target_level}']

            except Exception as e:
                error = f"[ERROR: {e}]"
                print(error)
                break
                time.sleep(1)


            predictions.append(
                {
                    "text_id": item["text_id"],
                    "simplified": simplified,
                }
            )

            time.sleep(0.1)  # Rate limiting

        # Save to submission folder
        output_path = submissions_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        print(f"Saved {len(predictions)} predictions to {output_path.relative_to(repo_root)}")
        return predictions
    return (generate_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Simplifications (Trial Data)

    Run the LLM on trial data to iterate on the prompt.
    """)
    return


@app.cell
def _(mo):
    run_trial_button = mo.ui.run_button(label="Generate Trial Predictions")
    run_trial_button
    return (run_trial_button,)


@app.cell
def _(
    SYSTEM_PROMPT,
    SimplificationResponse,
    build_prompt,
    generate_predictions,
    mo,
    model_selector,
    run_trial_button,
    temperature_slider,
    trial_data,
):
    trial_predictions = []

    if run_trial_button.value:
        with mo.status.spinner(title="Generating trial predictions...") as status:
            trial_predictions = generate_predictions(
                data=trial_data,
                model_name=model_selector.value,
                temperature=temperature_slider.value,
                system_prompt=SYSTEM_PROMPT,
                build_prompt_fn=build_prompt,
                response_format=SimplificationResponse,
                output_filename="dev_trial.jsonl",
                status_obj=status,
            )
    else:
        print("Click 'Generate Trial Predictions' to run")
    return (trial_predictions,)


@app.cell
def _(mo, trial_data, trial_predictions):
    pred_map_trial = {p["text_id"]: p["simplified"] for p in trial_predictions}
    preview_rows_trials = []
    for item_pred in trial_data[:5]:
        preview_rows_trials.append(
            {
                "text_id": item_pred["text_id"],
                "target": item_pred["target_cefr"],
                "original": item_pred["original"],
                "simplified": pred_map_trial.get(item_pred["text_id"], ""),
                "reference": item_pred["reference"],
            }
        )
    preview_table_trial = mo.ui.table(preview_rows_trials, label="Preview (first 5 examples)")


    preview_table_trial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Evaluation

    Execute the evaluation script on trial data.
    """)
    return


@app.cell
def _(mo):
    run_eval_button = mo.ui.run_button(label="Run Evaluation")
    run_eval_button
    return (run_eval_button,)


@app.cell
def _(mo, repo_root, run_eval_button, submissions_dir, trial_path):
    import subprocess
    import sys

    eval_output = ""

    if run_eval_button.value and (submissions_dir / "dev_trial.jsonl").exists():
        eval_script = repo_root / "evaluation" / "tsar2025_evaluation_script.py"

        env = {
            **dict(__import__("os").environ),
            "TSAR_GOLD_FILE": str(trial_path),
            "TSAR_SUBMISSIONS_DIR": str(submissions_dir.parent),
        }

        with mo.status.spinner(title="Running evaluation (this may take a while)..."):
            result = subprocess.run(
                [sys.executable, str(eval_script)],
                capture_output=True,
                text=True,
                cwd=str(repo_root / "evaluation"),
                env=env,
            )
            eval_output = result.stdout + "\n" + result.stderr

        mo.md(f"```\n{eval_output}\n```")
    elif run_eval_button.value:
        mo.md("*No submission file found. Generate and save predictions first.*")
    else:
        mo.md("*Click 'Run Evaluation' after generating and saving predictions.*")
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
    run_test_button = mo.ui.run_button(label="Generate Test Predictions")
    run_test_button
    return (run_test_button,)


@app.cell
def _(
    SYSTEM_PROMPT,
    SimplificationResponse,
    build_prompt,
    generate_predictions,
    mo,
    model_selector,
    run_test_button,
    temperature_slider,
    test_data,
):
    test_predictions = []

    if run_test_button.value and test_data:
        with mo.status.spinner(title="Generating test predictions...") as test_status:
            test_predictions = generate_predictions(
                data=test_data,
                model_name=model_selector.value,
                temperature=temperature_slider.value,
                system_prompt=SYSTEM_PROMPT,
                build_prompt_fn=build_prompt,
                response_format=SimplificationResponse,
                output_filename="final_test.jsonl",
                status_obj=test_status,
            )
    elif run_test_button.value:
        print("No test data available")
    else:
        print("Click 'Generate Test Predictions' to run on test set")
    return


if __name__ == "__main__":
    app.run()
