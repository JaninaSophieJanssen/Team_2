import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # GPT-2 Playground

    Experiment with GPT-2 text generation using Transformers.
    """)
    return


@app.cell
def _():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = "mps"
    model = model.to(device)
    return device, model, tokenizer


@app.cell
def _(mo):
    prompt_input = mo.ui.text_area(
        value="Once upon a time",
        label="Prompt",
        full_width=True,
        rows=3
    )
    return (prompt_input,)


@app.cell
def _(mo):
    max_length_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Max Length",
        show_value=True
    )

    temperature_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="Temperature",
        show_value=True
    )

    top_k_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=10,
        value=50,
        label="Top K",
        show_value=True
    )

    top_p_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.9,
        label="Top P",
        show_value=True
    )

    num_return_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        label="Number of Sequences",
        show_value=True
    )
    return (
        max_length_slider,
        num_return_slider,
        temperature_slider,
        top_k_slider,
        top_p_slider,
    )


@app.cell
def _(
    max_length_slider,
    mo,
    num_return_slider,
    temperature_slider,
    top_k_slider,
    top_p_slider,
):
    mo.hstack([
        mo.vstack([max_length_slider, temperature_slider]),
        mo.vstack([top_k_slider, top_p_slider, num_return_slider])
    ])
    return


@app.cell
def _(mo, prompt_input):
    generate_button = mo.ui.run_button(label="Generate", kind="success")
    mo.hstack([prompt_input, generate_button])
    return (generate_button,)


@app.cell
def _(
    device,
    generate_button,
    max_length_slider,
    model,
    num_return_slider,
    prompt_input,
    temperature_slider,
    tokenizer,
    top_k_slider,
    top_p_slider,
):
    if generate_button.value and prompt_input.value:
        input_ids = tokenizer.encode(prompt_input.value, return_tensors="pt").to(device)

        outputs = model.generate(
            input_ids,
            max_length=max_length_slider.value,
            temperature=temperature_slider.value,
            top_k=top_k_slider.value,
            top_p=top_p_slider.value,
            num_return_sequences=num_return_slider.value,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return (generated_texts,)


@app.cell
def _(generate_button, generated_texts, mo):
    if generate_button.value:
        results = "\n\n---\n\n".join([f"**Generation {i+1}:**\n\n{text}" for i, text in enumerate(generated_texts)])
        print(results)
        mo.md(f"""
        ## Generated Text

        {results}
        """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
