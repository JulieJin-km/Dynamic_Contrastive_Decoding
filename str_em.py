import numpy as np
import argparse
import json
import os
import regex
import string

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_str_em(qa_pairs):
    loc_acc = []
    for qa_pair in qa_pairs:
        loc_acc.append(exact_presence(qa_pair['answers'], qa_pair['prediction']))
    str_em = np.mean(loc_acc)

    return str_em

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default = None, help="the file to be evaluated")
    parser.add_argument("--d", type=str, default = None, help="Process the entire folder")
    parser.add_argument("--type", type=str, required=True, help="asqa, nq, tqa, popqa")

    args = parser.parse_args()

    if args.f is not None:
        with open(args.f) as f:
            datas = json.load(f)

        samples = datas.values()

        result = {}
        result['str_em'] = compute_str_em(samples)

        print(result)
        json.dump(result, open(args.f + ".score.json", "w"), indent=4)

    if args.d is not None:
        for filename in os.listdir(args.d):
            if "json" in filename and args.type in filename and "score" not in filename:
                with open(os.path.join(args.d, filename)) as f:
                    datas = json.load(f)

                samples = datas.values()

                result = {}
                result['str_em'] = compute_str_em(samples)

                print(filename, result)
                json.dump(result, open(os.path.join(args.d, filename + '.score.json'), "w"), indent=4)


if __name__ == "__main__":
    main()