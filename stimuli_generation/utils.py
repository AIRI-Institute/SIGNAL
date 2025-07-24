import random
from typing import Any, Dict, List, Optional, Tuple, Union

from pymorphy2 import MorphAnalyzer
from razdel import tokenize
from russian_tagsets import converters
from ruword_frequency import Frequency

def freq_group(frequency):
    if frequency > 10:
        return 1
    elif frequency > 4.5:
        return 2
    elif frequency > 1:
        return 3
    else:
        return 0
    

class TextIlliteracy:
    """Class for tokenizing text and changing variant of grammatical category
    in words of chosen parts of speech"""

    _tokens: List[str]

    def __init__(self, text: str) -> None:
        """Initializes an object"""
        self._text = text
        self._tokens = []

    def get_original_text(self) -> str:
        return self._text

    def tokenize_text(self) -> List[str]:
        """Tokenizes text"""
        pass

    def spoil_text(self, gram: str, postag_list: List[str]) -> str:
        """Changes grammatical markers to random inside the choosen category
        in all words of choosen parts of speech,
        if this grammatical category is relevant for such POS"""
        pass


class TextIlliteracyRus(TextIlliteracy):
    """Class tokenizes russian text and changes variant of grammatical category
    in words of chosen parts of speech."""

    _tokens: List[str]
    # https://pymorphy2.readthedocs.io/en/stable/user/grammemes.html#grammeme-docs

    __postags = [
        "NOUN",
        "ADJF",
        "ADJS",
        "COMP",
        "VERB",
        "INFN",
        "PRTF",
        "PRTS",
        "GRND",
        "NUMR",
        "ADVB",
        "NPRO",
        "PRED",
        "PREP",
        "CONJ",
        "PRCL",
        "INTJ",
    ]

    __grams = {
        "number": ["sing", "plur"],
        "case": [
            "nomn",
            "gent",
            "datv",
            "accs",
            "ablt",
            "loct",
            "voct",
            "gen2",
            "acc2",
            "loc2",
        ],
        "animacy": ["anim", "inan"],
        "gender": ["masc", "femn", "neut", "ms-f"],
        "aspect": ["perf", "impf"],
        "transitivity": ["tran", "intr"],
        "person": ["1per", "2per", "3per"],
        "tense": ["pres", "past", "futr"],
        "mood": ["indc", "impr"],
        "involvement": ["incl", "excl"],
        "voice": ["actv", "pssv"],
    }

    def tokenize_text(self) -> List[str]:
        """For russian:
        tokenizes text"""
        if self._tokens != []:
            tokens = self._tokens
        else:
            tokens_with_boundaries = list(tokenize(self._text))
            # получили список токенов с границами
            tokens = []  # список токенов с пробелами в нужных местах
            prev_tok_end = 0
            for substring in tokens_with_boundaries:
                if substring.start != prev_tok_end:
                    tokens.append(" ")
                tokens.append(substring.text)
                prev_tok_end = substring.stop
            self._tokens = tokens
        return tokens

    # def check_inflex(self, tok):
    #     inflex_tags = ['Fixd', 'Sgtm', 'Pltm']

    def change_form(self, new_gram_val, tok, gram):
        changed_tok = ""
        try:
            if "Fixd" not in tok.tag:
                if tok.inflect({new_gram_val}) != None:
                    changed_tok = tok.inflect({new_gram_val}).word
                    while changed_tok == tok.word:
                        new_gram_val = random.choice(TextIlliteracyRus.__grams[gram])
                        changed_tok = self.change_form(new_gram_val, tok, gram)
                # else:
                #     while tok.inflect({new_gram_val}) == None:
                #         new_gram_val = random.choice(TextIlliteracyRus.__grams[gram])
                #         changed_tok = self.change_form(new_gram_val, tok, gram)
        except RecursionError:
            changed_tok = ""
        return changed_tok

    def spoil_text(
        self,
        gram: str = "number",
        postag_list: List[str] = __postags,
        token_num=None,
        morph=MorphAnalyzer(),
    ) -> str:
        """For russian:
        changes grammatical markers to random inside the choosen category
        in all words of choosen parts of speech,
        if this grammatical category is relevant for such POS"""
        # берёт список частей речи и категорию,
        # которую у этих частей речи надо портить рандомными вариантами

        if self._tokens == []:
            self._tokens = self.tokenize_text()

        tokens = self._tokens

        changed_tok = ""

        if type(token_num) is int:
            tokens = self._text.split()
            tok = tokens[token_num]
            tok_analysed = morph.parse(tok)[0]
            if tok_analysed.tag.POS in postag_list and hasattr(tok_analysed.tag, gram):
                new_gram_val = random.choice(TextIlliteracyRus.__grams[gram])
                changed_tok = self.change_form(new_gram_val, tok_analysed, gram)
                if tok[0].isupper() and changed_tok != "":
                    changed_tok = changed_tok[0].upper() + changed_tok[1:]
        else:
            for tok in tokens[::-1]:
                tok_analysed = morph.parse(tok)[0]
                if tok_analysed.tag.POS in postag_list and hasattr(
                    tok_analysed.tag, gram
                ):
                    new_gram_val = random.choice(TextIlliteracyRus.__grams[gram])
                    changed_tok = self.change_form(new_gram_val, tok_analysed, gram)
                    if tok[0].isupper():
                        changed_tok = changed_tok[0].upper() + changed_tok[1:]
                    break

        idx = tokens.index(tok)
        if changed_tok and type(token_num) is int:
            changed_text = " ".join([*tokens[:idx], changed_tok, *tokens[idx + 1 :]])
        elif changed_tok:
            changed_text = "".join([*tokens[:idx], changed_tok, *tokens[idx + 1 :]])
        else:
            changed_text = "None"  # "".join([*tokens[:idx], tok, *tokens[idx + 1:]])
        return changed_text


class SemaBreak:
    def __init__(
        self,
        parser,
        vec_model,
        freq_top,
        freq_middle,
        freq_low,
        filter_words=list(),
        tag_converter=converters.converter("opencorpora-int", "ud20"),
        threshold=10,
        topn=100,
    ):
        self.parser = parser
        self.vec_model = vec_model
        self.to_ud = tag_converter
        self.threshold = threshold
        self.topn = topn

        self.freq = Frequency()
        self.freq.load()
        self.freq_top = freq_top
        self.freq_middle = freq_middle
        self.freq_low = freq_low

        self.known_parses = {}

        self.filter_words = filter_words

    def break_semantics_rusvect(self, tokenized, word):
        if word not in self.known_parses:
            self.known_parses[word] = self.parser.parse(word)
        morph_word = self.known_parses[word][0]

        if morph_word.tag.POS not in ["NOUN", "VERB", "ADJF"]:
            negreplace = "---"
        else:
            lemma = morph_word.normal_form
            gram_form = set(str(morph_word.tag).split()[1].split(","))
            if "nomn" in gram_form:
                gram_form.remove("nomn")
                gram_form.add("accs")
            pos_ud = self.to_ud(morph_word.tag.POS).split()[0]
            corp_word = f"{lemma}_{pos_ud}"
            freq_gr = freq_group(self.freq.ipm(lemma))

            negreplace = "---"
            break_flag = False
            generated = []
            num_candidates = 3
            if corp_word in self.vec_model:
                for negword, cos in self.vec_model.most_similar(
                    negative=[corp_word], topn=self.topn
                ):
                    neglem, negpos = negword.split("_")
                    negfreq_gr = freq_group(self.freq.ipm(neglem))

                    if negpos == pos_ud and negfreq_gr == freq_gr:
                        # print(negfreq_gr, freq_gr)
                        if neglem not in self.known_parses:
                            self.known_parses[neglem] = self.parser.parse(neglem)
                        negparses = self.known_parses[neglem]
                        for p in negparses:  # for every of pymorphy parses
                            if p.tag.POS and self.to_ud(p.tag.POS).split()[0] == pos_ud:
                                if p.tag.POS == "NOUN":
                                    if (
                                        p.tag.gender == morph_word.tag.gender
                                        and p.tag.animacy == morph_word.tag.animacy
                                    ):  # same gender for nouns
                                        # print('same gender')
                                        neginfl = p.inflect(gram_form)
                                        if neginfl:
                                            negreplace = (
                                                neginfl.word
                                            )  # put in the same grammatical form
                                            if (
                                                negreplace not in generated
                                                and len(negreplace) > 2
                                                and neginfl.normal_form
                                                not in self.filter_words
                                            ):
                                                generated.append(negreplace)
                                                if len(generated) == num_candidates:
                                                    break_flag = True
                                                    break
                                else:  # for other POS
                                    neginfl = p.inflect(gram_form)
                                    if neginfl:
                                        negreplace = neginfl.word
                                        if negreplace not in generated:
                                            generated.append(negreplace)
                                            if len(generated) == num_candidates:
                                                break_flag = True
                                                break
        i = tokenized.index(word)

        if negreplace == "---":
            return ""
        else:
            result = [
                " ".join(
                    tokenized[:i]
                    + [
                        negreplace,
                    ]
                    + tokenized[i + 1 :]
                )
                for negreplace in generated
            ]
            return result
