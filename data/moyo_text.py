import os
import spacy
import openai
from pathlib import Path
from tqdm import tqdm
from data.data_collection import write_gpt_response, get_gpt_sentences
import jinja2 as j2

openai.api_key = os.getenv("OPENAI_API_KEY")
nlp = spacy.load('en_core_web_sm')


def prepare_prompt(
    action_name: str, system_template_path: Path, user_template_path: Path
) -> tuple[str, str]:
    templates_dir = system_template_path.parent
    environment = j2.Environment(loader=j2.FileSystemLoader(templates_dir))
    system_template = environment.get_template(system_template_path.name)
    user_template = environment.get_template(user_template_path.name)
    sentences = {"sentences": action_name}
    system_prompt = system_template.render()
    user_prompt = user_template.render(sentences)
    return system_prompt, user_prompt


def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def process_humanml3d(corpus, save_path):
    for i in tqdm(range(len(corpus))):
        caption = corpus[i]
        start = 0.0
        end = 0.0
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
        with open(save_path, 'a+') as f:
            f.write('%s#%s#%s#%s\n'%(caption, tokens, start, end))


def main():
    dataset_dir = Path("/Users/orrav/Documents/Data/moyo_toolkit/data/mosh_smpl/val")
    
    gpt_dir = Path("/Users/orrav/Documents/Data/moyo_toolkit/data/mosh_smpl/mdm/gpt/val")
    gpt_dir.mkdir(exist_ok=True, parents=True)
    save_dir = Path("/Users/orrav/Documents/Data/moyo_toolkit/data/mosh_smpl/mdm/step_by_step/val")
    save_dir.mkdir(exist_ok=True, parents=True)

    system_template_path = Path("/Users/orrav/Documents/projects/human-feedback/data/templates/t2m_system.j2")
    user_template_path = Path("/Users/orrav/Documents/projects/human-feedback/data/templates/user_prompt.j2")
    run_gpt = False

    if run_gpt is True:
        for pkl_path in dataset_dir.rglob("*.pkl"):
            filename = pkl_path.stem
            output_path = gpt_dir / f"{filename}.json"
            action_name = filename.split("03596_")[-1].split("-")[0]
            action_name = action_name.replace("_", " ").replace("(", "").replace(")", "").rstrip()
            system_prompt, user_prompt = prepare_prompt(action_name, system_template_path, user_template_path)
            write_gpt_response(system_prompt, user_prompt, output_path, cache=True)

    for gpt_path in gpt_dir.rglob("*.json"):
        filename = gpt_path.stem
        save_path = save_dir / f"{filename}.txt"
        sentences = get_gpt_sentences(gpt_path)
        sentences = [x for x in sentences if len(x)>0]
        sentences = [x.split(f"{i+1}. ")[-1] for i, x in enumerate(sentences)]

        action_name = filename.split("03596_")[-1].split("-")[0]
        action_name = action_name.replace("_", " ").replace("(", "").replace(")", "").rstrip()
        sentences.append(f"a person does " + action_name)
        process_humanml3d(sentences, save_path)
        
            

if __name__ == "__main__":
    main()