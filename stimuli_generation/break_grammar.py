from argparse import ArgumentParser

import pandas as pd
from tqdm.auto import tqdm

from utils import TextIlliteracyRus

def break_sentence(sentence, position):
    gram_inc = TextIlliteracyRus(sentence)
    gram_inc_case = gram_inc.spoil_text("case", ["NOUN"], token_num=position)
    return gram_inc_case


def main():
    parser = ArgumentParser(description="The input and output datasets")
    parser.add_argument("--input", help="The input-csv file with congruent sentences")
    parser.add_argument("--output", default="semantics_results.csv", help="The output csv-file to save the results")
    parser.add_argument("--sentence_column", default="sentence", help="The name of the column with sentences in the input data")
    parser.add_argument("--position_column", default="position", help="The name of the column with positions of a noun to be replaced")

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df["gram_incongruent"] = df.apply(lambda x: break_sentence(x[args.sentence_column], x[args.position_column]), axis=1) 
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
