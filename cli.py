#
# JiWER - Jitsi Word Error Rate
#
# Copyright @ 2018 - present 8x8, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Provide a simple CLI wrapper for JiWER. The CLI does not support custom transforms.
"""

import click
import pathlib
import jiwer
from jiwer import AlignmentChunk
from jiwer import cer
from collections import namedtuple
from tabulate import tabulate
from utils import is_arabic

@click.command()
@click.option(
    "-r",
    "--reference",
    "reference_file",
    type=pathlib.Path,
    required=True,
    help="Path to new-line delimited text file of reference sentences.",
)
@click.option(
    "-h",
    "--hypothesis",
    "hypothesis_file",
    type=pathlib.Path,
    required=True,
    help="Path to new-line delimited text file of hypothesis sentences.",
)
@click.option(
    "--cer",
    "-c",
    "compute_cer",
    is_flag=True,
    default=False,
    help="Compute CER instead of WER.",
)
@click.option(
    "--align",
    "-a",
    "show_alignment",
    is_flag=True,
    default=False,
    help="Print alignment of each sentence.",
)
@click.option(
    "--global",
    "-g",
    "global_alignment",
    is_flag=True,
    default=False,
    help="Apply a global minimal alignment between reference and hypothesis sentences "
    "before computing the WER.",
)
def cli(
    reference_file: pathlib.Path,
    hypothesis_file: pathlib.Path,
    compute_cer: bool,
    show_alignment: bool,
    global_alignment: bool,
):
    """
    JiWER is a python tool for computing the word-error-rate of ASR systems. To use
    this CLI, store the reference and hypothesis sentences in a text file, where
    each sentence is delimited by a new-line character.
    The text files are expected to have an equal number of lines, unless the `-g` flag
    is used. The `-g` flag joins computation of the WER by doing a global minimal
    alignment.

    """
    with reference_file.open("r") as f:
        reference_sentences = [
            ln.strip() for ln in f.readlines() if len(ln.strip()) > 1
        ]

    with hypothesis_file.open("r") as f:
        hypothesis_sentences = [
            ln.strip() for ln in f.readlines() if len(ln.strip()) > 1
        ]
        for ln in f.readlines():
            if len(ln.strip()) < 1:
                print("##########ln.strip(): ",ln.strip())
 
    if not global_alignment and len(reference_sentences) != len(hypothesis_sentences):
        raise ValueError(
            f"Number of reference sentences "
            f"({len(reference_sentences)} in '{reference_file}') "
            f"and hypothesis sentences "
            f"({len(hypothesis_sentences)} in '{hypothesis_file}') "
            f"do not match! "
            f"Use the `--global` flag to compute the measures over a global alignment "
            f"of the reference and hypothesis sentences."
        )

    if compute_cer:
        if global_alignment:
            out = jiwer.process_characters(
                reference_sentences,
                hypothesis_sentences,
                reference_transform=jiwer.cer_contiguous,
                hypothesis_transform=jiwer.cer_contiguous,
            )
        else:
            out = jiwer.process_characters(
                reference_sentences,
                hypothesis_sentences,
            )
    else:
        if global_alignment:
            out = jiwer.process_words(
                reference_sentences,
                hypothesis_sentences,
                reference_transform=jiwer.wer_contiguous,
                hypothesis_transform=jiwer.wer_contiguous,
            )
        else:
            out = jiwer.process_words(reference_sentences, hypothesis_sentences)

    if show_alignment:
        print(jiwer.visualize_alignment(out, show_measures=True), end="")
        align(out)
    else:
        if compute_cer:
            print(out.cer)
        else:
            print(out.wer)




def align_word_output(word_output):
    SUB_TOKEN = "substitute"
    DEL_TOKEN = "delete"
    INS_TOKEN = "insert"
    
    references = word_output.references
    hypotheses = word_output.hypotheses
    alignments = word_output.alignments

    aligned_references = []
    aligned_hypotheses = []

    for ref, hyp, alignment in zip(references, hypotheses, alignments):
        aligned_ref = []
        aligned_hyp = []

        for chunk in alignment:
            if chunk.type == 'equal':
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    aligned_ref.append((ref[i], "equal"))
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    aligned_hyp.append((hyp[i], "equal"))

            elif chunk.type == 'substitute':
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    aligned_ref.append((ref[i], SUB_TOKEN))
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    aligned_hyp.append((hyp[i], SUB_TOKEN))

            elif chunk.type == 'delete':
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    aligned_ref.append((ref[i], "delete"))
                    aligned_hyp.append(("placeholder", DEL_TOKEN))

            elif chunk.type == 'insert':
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    aligned_ref.append(("placeholder", INS_TOKEN))
                    aligned_hyp.append((hyp[i], "insert"))

        aligned_references.append(aligned_ref)
        aligned_hypotheses.append(aligned_hyp)

    return aligned_references, aligned_hypotheses

def calculate_language_wer_and_cer_with_detailed_tables(aligned_refs, aligned_hyps):
    # Error counters for WER
    arabic_errors = {"sub": 0, "del": 0, "ins": 0}
    english_errors = {"sub": 0, "del": 0, "ins": 0}
    
    # Substitution breakdown
    substitutions = {"ar_ar": 0, "ar_en": 0, "en_en": 0, "en_ar": 0}
    
    # Deletions and insertions
    deletions = {"ar": 0, "en": 0}
    insertions = {"ar": 0, "en": 0}

    arabic_total_ref = 0
    english_total_ref = 0

    # Reference and hypothesis strings for CER
    arabic_refs = []
    arabic_hyps = []
    english_refs = []
    english_hyps = []

    # Iterate over aligned reference and hypothesis
    for ref_pair, hyp_pair in zip(aligned_refs, aligned_hyps):
        for ref_p, hyp_p in zip(ref_pair, hyp_pair):
            ref_word, ref_status = ref_p
            hyp_word, hyp_status = hyp_p

            # Detect the language of the reference word
            if ref_word != "placeholder":  # Skip placeholders used for alignment
                if is_arabic(ref_word):
                    arabic_total_ref += 1
                    arabic_refs.append(ref_word)
                    arabic_hyps.append(hyp_word if hyp_status != "insert" else "")
                else:
                    english_total_ref += 1
                    english_refs.append(ref_word)
                    english_hyps.append(hyp_word if hyp_status != "insert" else "")
    

            # Process errors
            if ref_status == "substitute":
                ref_lang = "ar" if is_arabic(ref_word) else "en"
                hyp_lang = "ar" if is_arabic(hyp_word) else "en"

                # Track general substitutions
                if ref_lang == "ar":
                    arabic_errors["sub"] += 1
                else:
                    english_errors["sub"] += 1
                
                # Track detailed substitution breakdown
                substitutions[f"{ref_lang}_{hyp_lang}"] += 1

            elif ref_status == "delete":
                ref_lang = "ar" if is_arabic(ref_word) else "en"
                deletions[ref_lang] += 1
                if ref_lang == "ar":
                    arabic_errors["del"] += 1
                else:
                    english_errors["del"] += 1

            elif hyp_status == "insert":

                hyp_lang = "ar" if is_arabic(hyp_word) else "en"
                ref_lang = "ar" if is_arabic(ref_word) else "en"
                insertions[hyp_lang] += 1
            

    

    # Calculate WER for Arabic and English
    arabic_total_errors = sum(arabic_errors.values())
    english_total_errors = sum(english_errors.values())

    arabic_wer = (arabic_total_errors / arabic_total_ref) * 100 if arabic_total_ref > 0 else 0
    english_wer = (english_total_errors / english_total_ref) * 100 if english_total_ref > 0 else 0

    # Calculate CER for Arabic and English
    arabic_refs = [x for x in arabic_refs if x != "placeholder"]
    english_refs = [x for x in english_refs if x != "placeholder"]
    arabic_cer = cer(" ".join(arabic_refs), " ".join(arabic_hyps)) * 100 if arabic_refs else 0
    english_cer = cer(" ".join(english_refs), " ".join(english_hyps)) * 100 if english_refs else 0

    # Data for tables
    arabic_data = [
        ["Errors", arabic_errors],
        ["Total Reference Words", arabic_total_ref],
        ["WER", f"{arabic_wer:.2f}%"],
    ]
    
    english_data = [
        ["Errors", english_errors],
        ["Total Reference Words", english_total_ref],
        ["WER", f"{english_wer:.2f}%"],
    ]
    
    # Print tables
    print("### Arabic ###")
    print(tabulate(arabic_data, headers=["Metric", "Value"], tablefmt="pretty"))
    
    print("\n### English ###")
    print(tabulate(english_data, headers=["Metric", "Value"], tablefmt="pretty"))
    print()
    # Substitutions Table
    print("### Substitutions Table ###")
    total_subs = sum(substitutions.values())
    subs_data = [
        [key.replace('_', ' to '), f"{(count / total_subs) * 100:.2f}%" if total_subs > 0 else "0.00%"]
        for key, count in substitutions.items()
    ]
    print(tabulate(subs_data, headers=["Type", "Percentage"], tablefmt="pretty"))
    
    # Deletions Table
    print("\n### Deletions Table ###")
    total_dels = sum(deletions.values())
    dels_data = [
        [f"{key} deletions", f"{(count / total_dels) * 100:.2f}%" if total_dels > 0 else "0.00%"]
        for key, count in deletions.items()
    ]
    print(tabulate(dels_data, headers=["Type", "Percentage"], tablefmt="pretty"))
    
    # Insertions Table
    print("\n### Insertions Table ###")
    total_ins = sum(insertions.values())
    ins_data = [
        [f"{key} insertions", f"{(count / total_ins) * 100:.2f}%" if total_ins > 0 else "0.00%"]
        for key, count in insertions.items()
    ]
    print(tabulate(ins_data, headers=["Type", "Percentage"], tablefmt="pretty"))

    return arabic_wer, english_wer, arabic_cer, english_cer


def align(word_output):

    references = [" ".join(sen) for sen in word_output.references]
    hypotheses = [" ".join(sen) for sen in word_output.hypotheses]

    # Overall Metrics Table
    wer = word_output.wer * 100
    cer = jiwer.cer(references, hypotheses) * 100
    overall_metrics = [
        ["WER", f"{wer:.2f}%"],
        ["CER", f"{cer:.2f}%"],
        ["Insertions", word_output.insertions],
        ["Deletions", word_output.deletions],
        ["Substitutions", word_output.substitutions],
    ]
    
    print("### Overall Metrics ###")
    print(tabulate(overall_metrics, headers=["Metric", "Value"], tablefmt="pretty"))
    print()
    aligned_refs, aligned_hyps = align_word_output(word_output)
    calculate_language_wer_and_cer_with_detailed_tables(aligned_refs, aligned_hyps)


if __name__ == "__main__":
    cli()
