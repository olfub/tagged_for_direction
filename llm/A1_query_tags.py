from pathlib import Path

#import torch
#torch.cuda.set_per_process_memory_fraction(0.85,0)
#torch.cuda.set_per_process_memory_fraction(0.85,1)

from tagging_causality.llm.datasets import bn_datasets

from tagging_causality.llm.llms import LLMProvider

llms = [
    #"claude-3-5-sonnet-20241022",
    "Meta-Llama-3.1-70B-Instruct",
    "Llama-3.3-70B-Instruct",
    "Qwen2.5-72B-Instruct",
    #"gpt-4-0613",
    #"gpt-4o-2024-08-06"
]

save_path = Path("../queries/raw")
skip_existing = True


key_dir = Path("../api_keys/")


system_prompt = "You are an expert in annotating variables to provide additional information that helps to support a causal discovery algorithm."

tagging_prompt_template = \
    "A tag is a single word or short phrase that describes a variable. Tags should be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. Tags will be used to identify causal directions between variables. Therefore, the individual sets of tags per variable should be discriminative enough to inform the algorithm. Variables can have multiple tags.\n" + \
    f'Consider the following variables: $VARIABLES.\n\n' + \
    "Please generate a list of tags that can be assigned to one or multiple variables. Generate the number of tags necessary to strike a good balance between expressivity and specificity. Avoid duplicate tags that contain the same set of variables. Reply with one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. The output should be machine parsable. For that reason, do not include any explanations or additional comments."

typing_prompt_template = \
    "A type is a single word or short phrase that describes a variable. Types should be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. Types will be used to identify causal directions between variables. Therefore, the individual types should be discriminative enough to inform the algorithm. Variables are assigned to a single type only.\n" + \
    f'Consider the following variables: $VARIABLES.\n\n' + \
    "Please generate a list of types that can be assigned to one or multiple variables. Generate the number of types necessary to strike a good balance between expressivity and specificity. Reply with one line per type, where each line starts with the name of the type, followed by a colon, and then a comma-separated list of variables that belong to that type. Make sure that no variable appears in more than one the lists. The output should be machine parsable. For that reason, do not include any explanations or additional comments."

templates = {
    "tag": tagging_prompt_template,
    "type": typing_prompt_template
}


def assemble_messages(prompt_template, variables):
    variables_str = ", ".join(variables)

    prompt = prompt_template.replace("$VARIABLES", variables_str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages


def main():
    provider = LLMProvider(key_dir=key_dir)

    save_path.mkdir(exist_ok=True, parents=True)

    use_desc = True  # use raw var names or human readable

    for llm_name in llms:
        llm = provider.get_interface(llm_name)
        for template_name, template_str in templates.items():
            for ds_name, ds_variables in bn_datasets.items():
                query_name = f"{llm_name}_{template_name}_{ds_name}_{use_desc}"
                save_loc = save_path / f"{query_name}.txt"
                print(">>", query_name)
                if skip_existing and save_loc.exists():
                    continue

                var_strs = ds_variables.values() if use_desc else ds_variables.keys()
                messages = assemble_messages(template_str, var_strs)
                response = llm.query(messages)
                #print("!!", response)

                #print(response)
                save_loc.write_text(response)
    print("done.")


if __name__ == "__main__":
    main()
