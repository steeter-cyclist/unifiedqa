import glob, json, argparse, os

class CycicEncoder():

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(self.data_dir):
            raise Exception("Data dir not found: {}".format(self.data_dir))
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def get_filename(self, name):
        path = os.path.join(self.data_dir, '*'+name)
        return glob.glob(path)[-1]
    
    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            lines = []
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    print("Could not parse line {} in file {}".format(line, input_file))
                    #raise
                lines.append(data)
            return lines

    def encode_and_write(self, questions, answers, split):
        fout = open(os.path.join(self.output_dir, split+".tsv"), "w")
        ans = open(os.path.join(self.output_dir, split+"_ans.tsv"), 'w')
        for question, answer in zip(questions, answers):
            # No contexts, so skip that part
            the_question = question['question']
            question_type = question["questionType"]
            if question_type == "true/false":
                answer_options = [question["answer_option0"], question["answer_option1"]]
            else:
                answer_options = [question["answer_option0"], question["answer_option1"], question["answer_option2"], question["answer_option3"], question["answer_option4"]]
            the_question = the_question.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
            if '?' not in the_question:
                the_question = the_question + '?'
                
            paragraph = ""
            for i, option in enumerate(answer_options):
                title = 'ABCDEFGH'[i]
                paragraph += f" ({title}) {option}"
            paragraph = paragraph.strip().replace("\t", "").replace("\n", "").replace("   ", " ").replace("  ", " ")
            correct_answer = answer_options[answer['correct_answer']]
            question_str = f"{the_question.strip()} \\n {paragraph.strip()}\t{correct_answer.strip()}\n"
            ans_json = json.dumps([correct_answer]) + "\n"
            fout.write(question_str)
            ans.write(ans_json)
            

    def encode_data_dir(self, split):
        questions_file = answers_file = ""
        if split == "train":
            questions_file = self.get_filename("training*questions.jsonl")
            answers_file = self.get_filename("training*labels.jsonl")
        elif split == "test":
            questions_file = self.get_filename("test*questions.jsonl")
            answers_file = self.get_filename("test*labels.jsonl")
        elif split == "dev":
            questions_file = self.get_filename("dev*questions.jsonl")
            answers_file = self.get_filename("dev*labels.jsonl")
        else:
            raise Exception("Unknown split: {}".format(split))
        questions = self._read_json(questions_file)
        answers = self._read_json(answers_file)
        self.encode_and_write(questions, answers, split)


def main():
    parser = argparse.ArgumentParser(description='Encode a CycIC dataset into the UnifiedQA format.')
    parser.add_argument('--data_dir', type=str, dest='data_dir', required=True, help="The directory to read the CycIC data from")
    parser.add_argument('--output_dir', type=str, dest='output_dir', required=True, help="The directory to write the encoded output to.")
    parser.add_argument('--split', type=str, dest='split', required=False, default='all', help="Which subset of the data to work on: 'train', 'test', 'dev', or 'all.'")
    args = parser.parse_args()
    encoder = CycicEncoder(args.data_dir, args.output_dir)
    if args.split == 'all':
        for split in ['train', 'dev', 'test']:
            encoder.encode_data_dir(split)
    else:
        encoder.encode_data_dir(args.split)

if __name__ == '__main__':
    main()
            
    
