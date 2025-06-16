from pathlib import Path

from datasets import bn_datasets


llms = [
    "claude-3-5-sonnet-20241022",
    "Meta-Llama-3.1-70B-Instruct",
    "Llama-3.3-70B-Instruct",
    "Qwen2.5-72B-Instruct",
    "gpt-4-0613",
    "gpt-4o-2024-08-06"
]

raw_path = Path("../queries/raw")
processed_path = Path("../queries/processed")

templates = ["tag", "type"]

#use_descs = [True, False]
use_descs = [True]

def main():
    processed_path.mkdir(exist_ok=True, parents=True)

    for use_desc in use_descs:
        for llm_name in llms:
            for template_name in templates:
                for ds_name, ds_variables in bn_datasets.items():
                    query_name = f"{llm_name}_{template_name}_{ds_name}_{use_desc}"
                    raw_loc = raw_path / f"{query_name}.txt"
                    processed_loc = processed_path / f"{query_name}.txt"
                    #print(f"processing {query_name}")

                    if not raw_loc.exists():
                        print(f"[{query_name}] Does not exist. Skipping.")
                        continue
                    raw_text = raw_loc.read_text().strip()

                    save_strs = list(ds_variables.keys())
                    if use_desc:
                        var_strs = list(ds_variables.values())
                    else:
                        var_strs = list(ds_variables.keys())

                    spellings = {var_name.lower():var_name for var_name in var_strs}

                    lines = raw_text.split("\n")
                    processed_lines = []
                    for line in lines:
                        tag_name, var_names = line.split(":")
                        tag_name = tag_name.strip()

                        processed_var_names = []
                        for var_name in var_names.split(","):
                            var_name_x = var_name.strip()
                            var_name = spellings.get(var_name_x.lower(), None)  # get correct capitalisation of variable (might be altered by some LLMs)
                            if var_name is None:
                                print(f"[{query_name}] {var_name_x} is not a valid variable {var_strs}. Skipping.")
                                continue
                            processed_var_name = save_strs[var_strs.index(var_name)]  # replace text var name with BN var name
                            processed_var_names.append(processed_var_name)

                        processed_line = f"{tag_name}:{','.join(processed_var_names)}"
                        processed_lines.append(processed_line)

                    processed_text = "\n".join(processed_lines)

                    processed_loc.write_text(processed_text)
    print("done.")


if __name__ == "__main__":
    main()
