import zipfile
import wget
from russian_tagsets import converters
import gensim
import pymorphy2
from tqdm.notebook import tqdm
from ruword_frequency import Frequency
import pandas as pd
from utils import SemaBreak
from argparse import ArgumentParser

    
def calculate_frequency():
    freq = Frequency()
    freq.load()
    freq_top = set(freq.iterate_words(10))
    freq_middle = set(freq.iterate_words(4.5)) - freq_top
    freq_low = set(freq.iterate_words(1)) - freq_top - freq_middle
    
    return freq_top, freq_middle, freq_low
    
    
def generate_sentences(model, parser, stimuli, ntop):
    br_sem_sents = []
    freq_top, freq_middle, freq_low = calculate_frequency()
    sem_breaker = SemaBreak(parser=parser, vec_model=model, topn=ntop,
                           freq_top=freq_top, freq_middle=freq_middle,
                           freq_low=freq_low)
    
    stimuli["sem_incongruent"] = stimuli.apply(lambda x: sem_breaker.break_semantics_rusvect(x["words"], x["Object"]), axis=1)
    return stimuli

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", help="The input-csv file with congruent sentences")
    parser.add_argument("--output", default="semantics_results.csv", help="The output csv-file to save the results")
    parser.add_argument("--sentence_column", default="congruent", help="The name of the column with sentences in the input data")
    parser.add_argument("--position_column", default="position", help="The name of the column with positions of a noun to be replaced")
    parser.add_argument("--model", default="model.bin", help="The file with word2vec model to be used")
    parser.add_argument("--ntop", default=400, help="The number of negative examples to generate")
    args = parser.parse_args()
    
    model = gensim.models.KeyedVectors.load_word2vec_format(args.model, binary=True)
    stimuli = pd.read_csv(args.input)
    parser = pymorphy2.MorphAnalyzer()
    stimuli["words"] = stimuli[args.sentence_column].apply(lambda x: x.split())
    stimuli["Object"] = stimuli.apply(lambda x: x["words"][x[args.position_column]], axis=1)
    broken_stimuli = generate_sentences(model, parser, stimuli, args.ntop)
    broken_stimuli.to_csv(args.output)

if __name__ == "__main__":
    main()