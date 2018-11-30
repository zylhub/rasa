from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import glob
import io
import logging
from tqdm import tqdm

from rasa_nlu import utils
from rasa_nlu.training_data import load_data, TrainingData, Message
from rasa_nlu.training_data.loading import _guess_format

logger = logging.getLogger(__name__)


class StartNextIntent(Exception):
    """Exception used to break out the annotation and start new intent."""
    pass


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Annotate new intents for an NLU model based on '
                    'collected conversations.')

    parser.add_argument('--language', required=True,
                        help="language")

    parser.add_argument('--conversations', required=True,
                        help="conversations")

    utils.add_logging_option_arguments(parser, default=logging.INFO)

    return parser


def load_conversations(conversations_path):
    events = []

    for ft in glob.glob(conversations_path + '/*.json'):
        events.append(utils.read_json_file(ft))

    msgs = [e.get("parse_data")
            for t in events
            for e in t.get("events", [])
            if e.get("event") == "user" and not e.get("text").startswith("/")]

    return msgs


def unique_msgs(msgs, max_length=1000):
    unique = {}
    for i in msgs:
        # makes sure we do not have to many duplicates that are only different
        # in their casing
        key = i.get("text", "").lower().strip()

        # removes line end punctuation
        if key and key[-1] in {"?", ".", "!", ",", "-"}:
            key = key[:-1]

        if key not in unique and len(key) < max_length:
            unique[key] = i
    return list(unique.values())


def get_docs(msgs, lm):
    for m in tqdm(msgs):
        m["doc"] = lm(m.get("text", ""))
    return msgs


def get_similarities(example_doc, other_docs):
    sims = []
    for d in tqdm(other_docs):
        sims.append((d, example_doc.similarity(d.get("doc"))))
    return sims


def print_similarities(sims, limit=200):
    for d, s in sims[:limit]:
        print("{:.2f}\t{}\t{}".format(s, d.get("intent"),
                                      d.get("text")))


def sorted_by_similarity(example_doc, other_docs):
    sims = get_similarities(example_doc, other_docs)
    return sorted(sims, key=lambda x: -x[1])


def _prepare_data(conversations_path, lm):
    conversations = load_conversations(conversations_path)
    return get_docs(unique_msgs(conversations), lm)


def _request_export_info():
    """Request file path to export data to."""
    from PyInquirer import prompt

    def validate_path(path):
        try:
            with io.open(path, "a", encoding="utf-8"):
                return True
        except Exception as e:
            return "Failed to open file. {}".format(e)

    # export training data and quit
    questions = [{
        "name": "export nlu",
        "type": "input",
        "message": "Export NLU examples to (if file exists, this "
                   "will merge with the previous examples)",
        "default": "nlu.md",
        "validate": validate_path
    }]

    answers = prompt(questions)
    if not answers:
        sys.exit()

    return answers["export nlu"]


def _write_nlu_to_file(export_nlu_path, messages):
    """Write the nlu data to the file paths."""
    from PyInquirer import prompt

    # noinspection PyBroadException
    try:
        previous_examples = load_data(export_nlu_path)

    except Exception:
        questions = [{"name": "export nlu",
                      "type": "input",
                      "message": "Could not load existing NLU data, please "
                                 "specify where to store NLU data learned in "
                                 "this session (this will overwrite any "
                                 "existing file)",
                      "default": "other-nlu.md"}]

        answers = prompt(questions)
        export_nlu_path = answers["export nlu"]
        previous_examples = TrainingData()

    nlu_data = previous_examples.merge(TrainingData(messages))

    with io.open(export_nlu_path, 'w', encoding="utf-8") as f:
        if _guess_format(export_nlu_path) in {"md", "unk"}:
            f.write(nlu_data.as_markdown())
        else:
            f.write(nlu_data.as_json())


def _ask_if_quit(messages):
    """Display the exit menu.

    Return `True` if the previous question should be retried."""
    from PyInquirer import prompt

    questions = [{
        "name": "abort",
        "type": "list",
        "message": "Do you want to stop?",
        "choices": [
            {
                "name": "Continue",
                "value": "continue",
            },
            {
                "name": "Next Intent",
                "value": "next_intent",
            },
            {
                "name": "Export & Quit",
                "value": "quit",
            },
        ]
    }]
    answers = prompt(questions)

    if not answers or answers["abort"] == "quit":
        # this is also the default answer if the user presses Ctrl-C
        nlu_path = _request_export_info()

        _write_nlu_to_file(nlu_path, messages)

        logger.info("Successfully wrote stories and NLU data")
        sys.exit()
    elif answers["abort"] == "continue":
        # in this case we will just return, and the original
        # question will get asked again
        return True
    elif answers["abort"] == "next_intent":
        raise StartNextIntent()


def _ask_questions(questions, messages, is_abort=lambda x: False):
    """Ask the user a question, if Ctrl-C is pressed provide user with menu."""
    from PyInquirer import prompt

    should_retry = True
    answers = {}

    while should_retry:
        answers = prompt(questions)
        if not answers or is_abort(answers):
            should_retry = _ask_if_quit(messages)
        else:
            should_retry = False
    return answers


def _enter_intent_name(messages):
    """Request a new intent name from the user."""

    questions = [{
        "name": "new_intent",
        "type": "confirm",
        "message": "Name of the new intent (Ctr-c to abort) with a very long message that the ccommandline needs to break so that we can properly read it othwerise we dont know hat to do!s:"
    }]

    answers = _ask_questions(questions, messages,
                             lambda a: not a["new_intent"])

    return answers["new_intent"]


def _enter_example_message_text(new_intent, messages):
    """Request a an example message for a new intent."""

    questions = [{
        "name": "example_message",
        "type": "input",
        "message": "Example message for the new intent '{}' (Ctr-c to abort):"
                   "".format(new_intent)
    }]

    answers = _ask_questions(questions, messages,
                             lambda a: not a["example_message"])

    return answers["example_message"]


def _annotate_example_with_intent(new_intent, example, messages):
    """Request a an example message for a new intent."""

    questions = [{
        "name": "annotation",
        "type": "list",
        "message": "Good example for '{}': '{}' ? (previously {}):"
                   "".format(new_intent,
                             example.get("text"),
                             example.get("intent", {}).get("name")),
        "choices": [
            {
                "name": "Yes",
                "value": "Yes",
            },
            {
                "name": "No",
                "value": "No",
            },
            {
                "name": "Skip",
                "value": "Skip",
            }
        ]
    }]

    answers = _ask_questions(questions, messages)

    if answers["annotation"] == "Yes":
        return True
    elif answers["annotation"] == "Skip":
        return None
    elif answers["annotation"] == "No":
        return False


def _print_help() -> None:
    """Print some initial help message for the user."""

    print("This script will help you create new intents and collect initial "
          "data for them. (press 'Ctr-c' to exit). ")


def interactive_annotation(conversations_path, language):
    import spacy

    logger.debug("Loading spacy language model")
    lm = spacy.load(language)
    logger.debug("Preparing existing data...")
    conversation_data = _prepare_data(conversations_path, lm)
    logger.info("Ready to go!")

    _print_help()

    collected_messages = []

    while True:
        next_intent = _enter_intent_name(collected_messages)

        example_message_text = _enter_example_message_text(next_intent,
                                                           collected_messages)

        collected_messages.append(
            Message.build(example_message_text , next_intent))

        print("Ok, let's go through the existing conversational data to find "
              "similar examples we can use as training data for this new "
              "intent.")

        sims = sorted_by_similarity(lm(example_message_text), conversation_data)

        try:
            for d, s in sims:
                response = _annotate_example_with_intent(next_intent, d,
                                                         collected_messages)
                if response:
                    collected_messages.append(
                        Message.build(d.get("text"), next_intent))
        except StartNextIntent:
            continue


if __name__ == '__main__':
    cmdline_args = create_argument_parser().parse_args()
    utils.configure_colored_logging(cmdline_args.loglevel)

    interactive_annotation(cmdline_args.conversations, cmdline_args.language)
