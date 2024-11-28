import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers import TopPLogitsWarper, LogitsProcessorList
from tqdm import tqdm
import argparse
import logging
from data_utils import Prompt, MDDataset
from str_em import normalize_answer, exact_presence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

CLOSED_BOOK_PROMPT = '''Question: {question}\nAnswer:'''
MD_RANDOM_PROMPT = prompt = '''Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). The search results are ordered randomly.

{search_results}

Question: {question}
Answer:'''
MD_PROMPT = '''Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). 

{search_results}

Question: {question}
Answer:'''


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_argument():
    parser = argparse.ArgumentParser()
    # environmental
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--model_path", type=str, default=0, help="the storage location of the model")
    # file
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, help="alce_asqa, nq, tqa, popqa")
    # prompt setting
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--doc_num", type=int, default=0)
    parser.add_argument("--prompt", type=str, help="closed, md or from file", required=True)
    parser.add_argument("--use_random", action="store_true", help="The documents are ordered randomly.")
    parser.add_argument("--special_location", type=int, default=-1)
    # generation config
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=4000)
    # evaluation setting
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--eval_during_infer", action="store_true")
    parser.add_argument("--metrics", type=str, default=None, help="asqa, strem")
    # structure
    parser.add_argument("--mode", type=str, default="standard", help="closed, full, single, cad, dvd")
    parser.add_argument("--closed_p", type=str, default=None, help="template for closed_book prompt")
    parser.add_argument("--alpha", type=float, default=0.2, help="the weight of cad")
    parser.add_argument("--beta", type=float, default=0.25, help="the weight of dvd")
    parser.add_argument("--gamma", type=float, default=0.2, help="the weight of dvd")
    parser.add_argument("--topk", type=int, default=10, help="the number of tokens with highest probability")
    parser.add_argument("--dynamic_weight", action="store_true")
    parser.add_argument("--retrieval", action="store_true", help="the selection criteria is based on retrieval")
    # debug
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.doc_num == 0:
        args.closed_book = True
        args.mode = "closed"
    else:
        args.closed_book = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    if args.special_location > 0:
        assert args.type == "nq_doc_num"
    if args.mode == "cad":
        args.postfix = "_" + str(args.alpha)
    elif args.mode == "dvd":
        if args.dynamic_weight:
            args.postfix = "_dw"
        else:
            args.postfix = "_b" + str(args.beta) + "_g" + str(args.gamma)
        if args.retrieval:
            args.postfix = args.postfix + "_retrieval"
    else:
        args.postfix = ""
    logger.info(args)
    return args


def evaluate_example(answers, prediction, metrics, qa_pairs=None):
    if metrics == "accuracy":
        normalized_prediction = normalize_answer(prediction)

        for ground_truth in answers:
            normalized_ground_truth = normalize_answer(ground_truth)
            if normalized_ground_truth.lower() in normalized_prediction.lower():
                return 1.0
        return 0.0
    elif metrics == "strem":
        return exact_presence(short_answers=answers, context=prediction)

    elif metrics == "asqa":
        # asqa: citations, qa, mauve, str_em, str_hit
        # QAMPARI: citations
        # eli5: claims_nli, mauve
        loc_acc = []
        for qa_pair in qa_pairs:
            loc_acc.append(exact_presence(qa_pair['short_answers'], prediction))
        str_em = np.mean(loc_acc)
        str_hit = int(np.mean(loc_acc) == 1)

        return (str_em, str_hit)



def mode_generate(example, model, args, P, CP, tokenizer,):
    processors = LogitsProcessorList()
    processors.append(TopPLogitsWarper(min(0.95, args.top_p)))
    closed_prompt = CP.apply([example], use_random=args.use_random, with_answer=False)
    full_prompt = P.apply([example], use_random=args.use_random, with_answer=False, doc_cluster=args.doc_num)
    doc_prompt = P.apply([example], use_random=args.use_random, with_answer=False, doc_cluster=1)
    batch = [closed_prompt] + full_prompt + doc_prompt
    if args.debug:
        print(len(batch))
        for i in range(len(batch)):
            print(batch[i])

    inputs = tokenizer(batch, padding='longest', return_tensors='pt').to(args.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    prompt_len = len(input_ids[0])
    logger.info(f"The batch shape is {input_ids.shape}")

    past_key_values = None
    n = input_ids.shape[0]
    model_output = []
    max_tokens = min(args.max_new_tokens, args.max_length - prompt_len)
    tokens = None
    tau = args.temperature
    '''
    if args.debug:
        print("input_ids:", input_ids)
    '''
    for i in range(max_tokens):
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        # ===== 核心部分，对logits进行调整 =====

        if args.debug:
            print("output:", outputs.logits)
            print(outputs.logits.shape)

        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        logits = processors(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)

        logits_uncond = logits[0]
        logits_full = logits[1]
        logits_rebest = logits[2]
        if args.mode == "closed":
            final_logit = logits_uncond
        elif args.mode == "full":
            final_logit = logits_full
        elif args.mode == "single":
            final_logit = logits_rebest
        elif args.mode == "cad":
            logits_merged = (1 + args.alpha) * logits_full - args.alpha * logits_uncond
            final_logit = torch.where(logits_uncond > -100, logits_merged, logits_full)
        elif args.mode == "dvd":
            temp_logit = logits[0]
            probas = torch.nn.functional.softmax(temp_logit[None] / tau, dim=-1)
            values, indices = torch.topk(probas, args.topk, largest=True)
            V = temp_logit[indices]
            cad_entropy = -(V.exp() * V.clip(-100, 0)).sum(dim=-1).item()
            values, _ = torch.topk(probas, 2, largest=True, sorted=True)
            cad_sub = values[0][0] - values[0][1]

            temp_logit = logits[1]
            probas = torch.nn.functional.softmax(temp_logit[None] / tau, dim=-1)
            values, indices = torch.topk(probas, args.topk, largest=True)
            V = temp_logit[indices]
            full_entropy = -(V.exp() * V.clip(-100, 0)).sum(dim=-1).item()
            values, _ = torch.topk(probas, 2, largest=True, sorted=True)
            full_sub = values[0][0] - values[0][1]


            logits_max = logits[2]
            logits_min = logits[n - 1]


            if not args.retrieval:
                maxent = -0.01
                minent = 50.0
                for i in range(2, n):
                    temp_logit = logits[i]
                    probas = torch.nn.functional.softmax(temp_logit[None] / tau, dim=-1)
                    values, indices = torch.topk(probas, args.topk, largest=True)
                    V = temp_logit[indices]
                    entropy = -(V.exp() * V.clip(-100, 0)).sum(dim=-1).item()
                    if entropy > maxent:
                        maxent = entropy
                        logits_min = temp_logit
                    if entropy < minent:
                        minent = entropy
                        logits_max = temp_logit

            if args.dynamic_weight:
                    # dynamic_weight
                temp_logit = logits_max
                probas = torch.nn.functional.softmax(temp_logit[None] / tau, dim=-1)
                values, _ = torch.topk(probas, 2, largest=True, sorted=True)
                max_sub = values[0][0] - values[0][1]

                temp_logit = logits_min
                probas = torch.nn.functional.softmax(temp_logit[None] / tau, dim=-1)
                values, _ = torch.topk(probas, 2, largest=True, sorted=True)
                min_sub = values[0][0] - values[0][1]

                gamma = max(max_sub - min_sub, 0)  # max
                beta = max(full_sub - cad_sub, 0)  # full

                if cad_entropy * 10 < full_entropy:
                    logits_merged = logits_uncond + gamma * logits_max - gamma * logits_min
                    final_logit = torch.where(logits_min > -100, logits_merged, logits_full)
                else:
                    # CAD + documents
                    logits_merged = (1 + beta) * logits_full - beta * logits_uncond + gamma * logits_max - gamma * logits_min
                    final_logit = torch.where(logits_uncond > -100, logits_merged, logits_full)
                    final_logit = torch.where(logits_min > -100, final_logit, logits_full)

            else:
                # static beta and gamma
                if cad_entropy * 10 < full_entropy:
                    logits_merged = logits_full + args.gamma * logits_max - args.gamma * logits_min
                    final_logit = torch.where(logits_min > -100, logits_merged, logits_full)
                else:
                    # CAD + documents
                    logits_merged = (1 + args.beta) * logits_full - args.beta * logits_uncond + args.gamma * logits_max - args.gamma * logits_min
                    final_logit = torch.where(logits_uncond > -100, logits_merged, logits_full)
                    final_logit = torch.where(logits_min > -100, final_logit, logits_full)



        probas = torch.nn.functional.softmax(final_logit[None] / tau, dim=-1)
        full_probas = torch.nn.functional.softmax(logits_full[None] / tau, dim=-1)
        probas = torch.where(torch.isnan(probas), full_probas, probas)
        try:
            next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        except RuntimeError as e:
            print(e)
            print("Error!")
            print(entropy)
            print(logits)
            print(probas)
            print(tokenizer.decode(tokens[0]) if tokens is not None else 'None')
            break
        if next_tokens[0] == tokenizer.eos_token_id:
            break

        # print(next_tokens)
        if tokens is None:
            tokens = next_tokens[:, None]
        else:
            tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        # print(tokens)
        ret = tokenizer.batch_decode(next_tokens)
        if ret[0] == '\n':
            break
        model_output.append(ret[0])

        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=args.device)], dim=-1)

    result = {}
    result['prompt'] = batch
    result['question'] = example.question
    result['answers'] = example.answers
    result['token_len_prompt'] = prompt_len,
    result['prediction'] = tokenizer.decode(tokens[0]) if tokens is not None else ''
    # print(' '.join(model_output))
    # print(tokens)
    # print(tokenizer.decode(tokens[0]))

    return result



def main():
    args = get_argument()
    set_seed(args)


    if args.do_inference:
        logger.info(f"The model path is {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True, trust_remote_code=True)
        tokenizer.padding_side = "left"
        torch_type = torch.float16
        tokenizer.pad_token = tokenizer.eos_token


        logger.info(f"Begin to process the data from input_file {args.input_file}, type is {args.type}")
        if args.type in ["nq_doc_num", "nq", "tqa", "alce_asqa", "popqa"]:
            eval_dataset = MDDataset(args.input_file, args.type)
        else:
            raise NotImplementedError

        logger.info(f"The chosen prompt is {args.prompt} with doc_num is {args.doc_num}.")
        if args.use_random:
            logger.info("The document is ordered randomly.")
        else:
            logger.info("The document is not ordered randomly.")
        if args.prompt == "closed":
            P = Prompt("str", CLOSED_BOOK_PROMPT, closed_book=True, ndoc=0)
            CP = P
        elif args.prompt == "md":
            if args.use_random:
                P = Prompt("str", MD_RANDOM_PROMPT, closed_book=False, ndoc=args.doc_num)
            else:
                P = Prompt("str", MD_PROMPT, closed_book=False, ndoc=args.doc_num)
            if args.closed_p is not None:
                CP = Prompt("str", CLOSED_BOOK_PROMPT, closed_book=True, ndoc=0)
            else:
                CP = None
        else:
            # for alce
            P = Prompt("json", args.prompt, closed_book=args.closed_book, ndoc=args.doc_num)
            if args.closed_p is not None:
                CP = Prompt("json", args.closed_p, closed_book=True, ndoc=0)
            else:
                CP = None


        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_type, device_map='auto',
                                                         use_auth_token=True, trust_remote_code=True)


        model.eval()

        datas = {}
        if args.eval_during_infer:
            logger.info(f"Evaluate during inference with metric {args.metrics}.")
            assert args.metrics in ["accuracy", "strem", "asqa"]
            scores = {
                "accuracy": [],
                "str_em": [],
                "asqa_str_em": [],
                "asqa_str_em_hit": []
            }

        logger.info(f"The generation mode is {args.mode}")
        if args.mode == "cad":
            logger.info(f"The weight alpha is {args.alpha}")
        if args.mode == "dvd":
            logger.info(f"TopK is {args.topk}.")
            if args.retrieval:
                logger.info("The selection criteria is based on retrieval.")
            if args.dynamic_weight:
                logger.info("The weights are settled dynamically during the generation. ")
            else:
                logger.info(f"The weights are static. Beta is {args.beta}. Gamma is {args.gamma}.")
        with torch.inference_mode():
            for example in tqdm(eval_dataset, total=len(eval_dataset)):
                qid = example.qas_id
                datas[qid] = {}
                datas[qid].update(mode_generate(example, model, args, P, CP, tokenizer))
                if example.qa_pairs is not None:
                    datas[qid]['qa_pairs'] = example.qa_pairs
                if example.annotations is not None:
                    datas[qid]['annotations'] = example.annotations
                if example.claims is not None:
                    datas[qid]['claims'] = example.claims
                if args.eval_during_infer:
                    score = evaluate_example(example.answers, datas[qid]['prediction'], args.metrics, example.qa_pairs)
                    if args.metrics == "accuracy":
                        scores['accuracy'].append(score)
                    elif args.metrics == "strem":
                        scores["str_em"].append(score)
                    elif args.metrics == "asqa":
                        scores['asqa_str_em'].append(score[0])
                        scores["asqa_str_em_hit"].append(score[1])

                if len(datas) < 10:
                    if isinstance(datas[qid]['prompt'], list):
                        for p in datas[qid]['prompt']:
                            print(p)
                    else:
                        print(datas[qid]['prompt'])
                    if "token_len_prompt" in datas[qid]:
                        print(datas[qid]['token_len_prompt'])
                    print(example.answers)
                    print(datas[qid]['prediction'])
                else:
                    if args.debug and len(datas) == 50:
                        break

        if args.eval_during_infer:
            for key in scores.keys():
                if len(scores[key]) > 0:
                    scores[key] = (sum(scores[key]) / len(scores[key]))
                    print(key, scores[key])

        output_file = args.model_path.split('/')[-1] + '_' + eval_dataset.type + '_' + str(
            args.doc_num) + 'doc_' + str(args.shot) + 'shot_' + args.mode + args.postfix + '.json'
        logger.info(f"The result is saved to {args.output_dir}/{output_file}")
        with open(os.path.join(args.output_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(datas, f, indent=4)
            f.close()



if __name__ == '__main__':
    main()
