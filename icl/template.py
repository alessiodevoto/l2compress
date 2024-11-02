import datasets
import random
import json
import logging
from typing import Dict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TASKS = dict()


class Template:
    task = None

    def __init__(self, input_key, output_key, input_key_verb, output_key_verb,
                 intra_sep, inter_sep):
        self.input_key = input_key
        self.output_key = output_key
        self.input_key_verb = input_key_verb
        self.output_key_verb = output_key_verb
        self.intra_sep = intra_sep
        self.inter_sep = inter_sep

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        TASKS[cls.task] = cls

    def normalise_input(self, item_input, *args, **kwargs):
        return item_input.strip()

    def output_mapping(self, *args, **kwargs):
        raise NotImplementedError

    def verbalize_item(self, item, is_test=False):
        input_text = self.normalise_input(item[self.input_key])
        if is_test:
            return f"{self.input_key_verb}{input_text}{self.intra_sep}{self.output_key_verb}".strip()
        else:
            output_text = self.output_mapping(item[self.output_key])
            return f"{self.input_key_verb}{input_text}{self.intra_sep}{self.output_key_verb}{output_text}"

    def verbal_prompt(self, input_example, demonstrations):
        prompt = ""
        for demonstration in demonstrations:
            prompt = prompt + self.verbalize_item(demonstration, is_test=False) + self.inter_sep
        prompt = prompt + self.verbalize_item(input_example, is_test=True)
        return prompt

    def jsonline_prompt(self, input_example, demonstrations):
        self.input_key_verb = self.input_key_verb.strip()
        self.output_key_verb = self.output_key_verb.strip()
        if self.input_key_verb.endswith(":"):
            self.input_key_verb = self.input_key_verb[:-1]
        if self.output_key_verb.endswith(":"):
            self.output_key_verb = self.output_key_verb[:-1]

        prompt = ""
        for demonstration in demonstrations:
            input_text = self.normalise_input(demonstration[self.input_key])
            output_text = self.output_mapping(demonstration[self.output_key])
            dict_demonstration = {
                self.input_key_verb.strip(): input_text,
                self.output_key_verb.strip(): output_text
            }
            prompt = prompt + json.dumps(dict_demonstration) + "\n"

        input_text = self.normalise_input(input_example[self.input_key])
        dict_input_example = {
            self.input_key_verb.strip(): input_text,
            self.output_key_verb.strip(): "###-yuzhaouoe-###"
        }  # the output is a placeholder
        verb_input_example = json.dumps(dict_input_example)
        remove_part = ' "###-yuzhaouoe-###"}'  # remove the output. it starts with a blank.
        remove_index = verb_input_example.index(remove_part)
        assert remove_index > 0
        verb_input_example = verb_input_example[:remove_index]

        prompt = prompt + verb_input_example
        return prompt

    def process_jsonline_output(self, batch_input_ids, batch_pred_ids, tokenizer):
        assert len(batch_input_ids) == len(batch_pred_ids)
        if not isinstance(batch_input_ids, list):
            batch_input_ids = batch_input_ids.tolist()
        if not isinstance(batch_pred_ids, list):
            batch_pred_ids = batch_pred_ids.tolist()
        preds = []
        for input_ids, pred_ids in zip(batch_input_ids, batch_pred_ids):
            seq_ids = input_ids + pred_ids
            seq = tokenizer.decode(seq_ids, skip_special_tokens=True)
            seq = seq.strip().split("\n")[-1]
            try:
                pred = json.loads(seq)
                pred = pred[self.output_key_verb]
            except Exception:
                pred = "###WrongJsonFormat###"
            preds.append(pred)
        return preds


class SST2(Template):
    task = "sst2"

    def __init__(self, input_key_verb, output_key_verb, intra_sep, inter_sep,
                 mapping_type, output_mapping_dict=None):
        # Init dataset
        data = datasets.load_dataset("SetFit/sst2")
        input_key = "text"
        output_key = "label"
        self.train_data = [item for item in data["train"]]
        self.test_data = [item for item in data["test"]]
        self.dev_data = [item for item in data["validation"]]

        # Init output mapping
        self.predefined_mapping_dicts = {
            "regular": {0: "negative", 1: "positive"},
            "number": {0: "0", 1: "1"},
            "alphabet": {0: "A", 1: "B"},
            "unrelated": {0: "foo", 1: "bar"},
            "flipped": {0: "positive", 1: "negative"},
        }
        self.mapping_type = None
        self.output_mapping_dict = None
        self.set_mapping_dict(output_mapping_dict=output_mapping_dict, mapping_type=mapping_type)

        super(SST2, self).__init__(input_key, output_key, input_key_verb,
                                   output_key_verb, intra_sep, inter_sep)

    def set_mapping_dict(self, output_mapping_dict=None, mapping_type=None):

        self.mapping_type = mapping_type
        if mapping_type == "self-define":
            assert output_mapping_dict is not None
            self.output_mapping_dict = output_mapping_dict
        else:
            self.output_mapping_dict = self.predefined_mapping_dicts[mapping_type]

    def output_mapping(self, x):
        return self.output_mapping_dict[x]


def get_task(task, **kwargs):
    return TASKS[task](**kwargs)


def debug():
    # task = SST2(
    #     input_key_verb="text: ",
    #     output_key_verb="label: ",
    #     intra_sep="\n",
    #     inter_sep="\n",
    #     mapping_type="unrelated",
    #     output_mapping_dict=None,
    # )
    task_kwargs = {
        "input_key_verb": "text: ",
        "output_key_verb": "label: ",
        "intra_sep": "\n",
        "inter_sep": "\n\n",
        "mapping_type": "unrelated",
        "output_mapping_dict": None,
    }
    task = get_task("sst2", **task_kwargs)

    demonstrations = random.sample(task.train_data, 8)
    for item in task.test_data:
        prompt = task.verbal_prompt(item, demonstrations)
        # prompt = task.jsonline_prompt(item, demonstrations)
        print(prompt)
        break


if __name__ == '__main__':
    debug()
