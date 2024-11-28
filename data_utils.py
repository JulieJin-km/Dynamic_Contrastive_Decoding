import json
import random
from xopen import xopen

class Document:
    def __init__(self,text,source="retriever" , title = None, score = None, id = None, hasanswer = None, isgold = None,
                 original_retrieval_index=None, summary = None):
        self.source = source
        self.text = text
        self.title = title
        self.score = score
        self.id = id
        self.hasanswer = hasanswer
        self.isgold = isgold
        self.original_retrieval_index = original_retrieval_index
        self.summary = summary

class MDExample:
    def __init__(
            self,
            qas_id,
            question_text,
            answers=[],
            documents=[],
            qa_pairs=None,
            annotations=None,
            claims=None,
    ):
        self.qas_id = qas_id
        self.question = question_text
        self.answers = answers
        self.documents = documents
        self.doc_size = len(documents)
        self.qa_pairs = qa_pairs # for evaluation in alce-asqa
        self.annotations = annotations
        self.claims = claims


class MDDataset(object):
    def __init__(self,filename, type):
        self.filename = filename
        self.type = type
        self.examples = self.read_data(filename, type)
        self.size = len(self.examples)
        self.min_doc_num = min([e.doc_size for e in self.examples])
        self.max_doc_num = max([e.doc_size for e in self.examples])

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return self.size

    def read_data(self, filename, type):
        examples = []
        if type == "nq_doc_num":
            with xopen(filename) as fin:
                for i,line in enumerate(fin):
                    input_example = json.loads(line)
                    question = input_example["question"]
                    answers = input_example["answers"]
                    documents = []
                    for ctx in input_example["ctxs"]:
                        documents.append(Document(text=ctx["text"], title=ctx["title"], score=ctx.get("score"), id=ctx.get("id"),
                                                  hasanswer=ctx["hasanswer"], isgold=ctx["isgold"],
                                                  original_retrieval_index=ctx.get("original_retrieval_index")))
                    examples.append(MDExample(qas_id=str(i),question_text=question, answers=answers,documents=documents))
            return examples

        elif type == "tqa" or type == "nq" or type == "popqa":
            with open(filename, 'r') as fin:
                datas = json.load(fin)
                fin.close()
            for i, input_example in enumerate(datas):
                question = input_example['question']
                answers = input_example["answers"]
                documents = []
                for ctx in input_example['ctxs']:
                    documents.append(Document(id=ctx['id'],title=ctx['title'],text=ctx['text']))
                examples.append(MDExample(qas_id=str(i), question_text=question, answers=answers, documents=documents))
            return examples
        elif "alce" in type:
            with open(filename,'r') as fin:
                datas = json.load(fin)
                fin.close()
            if "demo" in type:
                datas = datas['demos']
            for i, item in enumerate(datas):
                id = item.get('sample_id',None)
                if id == None:
                    id = item.get("id", None)
                    if id == None:
                        id = str(i)
                question = item['question']
                if isinstance(item["answer"], list):
                    answers = item['answer']
                else:
                    answers = [item['answer']]
                documents = []
                for doc in item['docs']:
                    documents.append(Document(text=doc['text'],id=doc.get("id",None),title=doc['title'],
                                              score=doc.get('score',None),hasanswer=doc.get('has_answer',None),
                                              summary=doc.get("summary",None)))
                examples.append(MDExample(qas_id=id,question_text=question,answers=answers,documents=documents,
                                          qa_pairs=item.get("qa_pairs"),annotations=item.get("annotations"),
                                          claims=item.get("claims")))
            return examples
        else:
            raise NotImplementedError



class Prompt(object):
    def __init__(self, type, prompt, closed_book = True, ndoc = 0, tokenizer = None, max_tokenlen = None):
        self.type = type
        self.sep = '\n\n'
        self.instruction = None
        self.closed_book = closed_book
        if self.type == "str":
            self.prompt = prompt
            self.instruction = prompt
        else:
            self.prompt = self.read_prompt(prompt)
        if self.closed_book:
            self.ndoc = 0
        else:
            self.ndoc = ndoc
        self.tokenizer = tokenizer
        self.max_tokenlen = max_tokenlen

    def read_prompt(self,prompt):
        with open("prompts/" + prompt, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            f.close()
        prompt = datas['demo_prompt'].replace("{INST}", datas["instruction"]).replace("{Q}", "{question}").replace(" {A}","")
        self.sep = datas['demo_sep']
        self.instruction = datas['instruction']
        if self.closed_book:
            return prompt
        else:
            prompt = prompt.replace("{D}","{search_results}")
            return prompt

    def apply(self, examples,use_random = False, with_answer = False, special = False, doc_cluster = -1):
        datas = []
        for example in examples:
            if self.closed_book:
                if with_answer:
                    if use_random:
                        datas.append(self.prompt.format(question=example.question) + ' ' + random.sample(example.answers,1))
                    else:
                        datas.append(self.prompt.format(question=example.question) + ' ' + example.answers[0])
                else:
                    datas.append(self.prompt.format(question=example.question))
            else:
                formatted_documents = []
                filtered_documents = example.documents
                if self.ndoc != -1:
                    if use_random:
                        if special != -1:
                            (original_gold_index,) = [idx for idx, doc in enumerate(filtered_documents) if doc.isgold is True]
                            original_gold_document = filtered_documents[original_gold_index]
                            distractors = [doc for doc in filtered_documents if doc.isgold is False]
                            random.shuffle(distractors)
                            distractors.insert(original_gold_index, original_gold_document)
                            filtered_documents = distractors
                        else:
                            filtered_documents = random.sample(filtered_documents,self.ndoc)
                    else:
                        filtered_documents = filtered_documents[:self.ndoc]
                for document_index, document in enumerate(filtered_documents):
                    formatted_documents.append(f"Document [{document_index + 1}](Title: {document.title}) {document.text}")
                if doc_cluster == -1:
                    if with_answer:
                        if use_random:
                            datas.append(self.prompt.format(question=example.question, search_results="\n".join(
                                formatted_documents)) + ' ' + random.sample(example.answers, 1))
                        else:
                            datas.append(self.prompt.format(question=example.question,
                                                            search_results="\n".join(formatted_documents)) + ' ' +
                                         example.answers[0])
                    else:
                        datas.append(
                            self.prompt.format(question=example.question, search_results="\n".join(formatted_documents)))
                else:
                    # no context
                    # datas.append(self.prompt.format(question=example.question))
                    # with every cluster context
                    i = 0
                    while i < len(formatted_documents):
                        datas.append(
                            self.prompt.format(question=example.question,
                                               search_results="\n".join(formatted_documents[i:i + doc_cluster])))
                        i = i + doc_cluster
                    return datas

        return self.sep.join(datas)

    def construct_context(self, example, doc_cluster=1):
        if self.closed_book:
            return [self.instruction]
        else:
            filtered_documents = example.documents[:self.ndoc]
            formatted_documents = []
            for document_index, document in enumerate(filtered_documents):
                formatted_documents.append(f"Document [{document_index + 1}](Title: {document.title}) {document.text}")

            contexts = []
            i = 0
            while i < len(formatted_documents):
                contexts.append(self.instruction + '\n\n' + "\n".join(formatted_documents[i:i + doc_cluster]) + '\n')
                i = i + doc_cluster
            return contexts


