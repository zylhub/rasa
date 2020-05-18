import itertools
import os
import logging
import numpy as np
from collections import defaultdict, namedtuple
from tqdm import tqdm
from typing import (
    Iterable,
    Iterator,
    Tuple,
    List,
    Set,
    Optional,
    Text,
    Union,
    Dict,
    Any,
)
import rasa.utils.plotting as plot_utils
import rasa.utils.io as io_utils

from rasa.constants import TEST_DATA_FILE, TRAIN_DATA_FILE, NLG_DATA_FILE
from rasa.nlu.constants import (
    DEFAULT_OPEN_UTTERANCE_TYPE,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    OPEN_UTTERANCE_PREDICTION_KEY,
    EXTRACTOR,
    PRETRAINED_EXTRACTORS,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    INTENT,
)
from rasa.model import get_model
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer, TrainingData
from rasa.nlu.components import Component
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.utils.tensorflow.constants import ENTITY_RECOGNITION

logger = logging.getLogger(__name__)

# Exclude 'EmbeddingIntentClassifier' and 'ResponseSelector' as their super class
# performs entity extraction but those two classifiers don't
ENTITY_PROCESSORS = {
    "EntitySynonymMapper",
    "EmbeddingIntentClassifier",
    "ResponseSelector",
}

CVEvaluationResult = namedtuple("Results", "train test")

NO_ENTITY = "no_entity"

IntentEvaluationResult = namedtuple(
    "IntentEvaluationResult", "intent_target intent_prediction message confidence"
)

ResponseSelectionEvaluationResult = namedtuple(
    "ResponseSelectionEvaluationResult",
    "intent_target response_target response_prediction message confidence",
)

EntityEvaluationResult = namedtuple(
    "EntityEvaluationResult", "entity_targets entity_predictions tokens message"
)

IntentMetrics = Dict[Text, List[float]]
EntityMetrics = Dict[Text, Dict[Text, List[float]]]
ResponseSelectionMetrics = Dict[Text, List[float]]


def log_evaluation_table(
    report: Text, precision: float, f1: float, accuracy: float
) -> None:  # pragma: no cover
    """Log the sklearn evaluation metrics."""
    logger.info(f"F1-Score:  {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Accuracy:  {accuracy}")
    logger.info(f"Classification report: \n{report}")


def get_evaluation_metrics(
    targets: Iterable[Any],
    predictions: Iterable[Any],
    output_dict: bool = False,
    exclude_label: Text = None,
) -> Tuple[Union[Text, Dict[Text, Dict[Text, float]]], float, float, float]:
    """Compute the f1, precision, accuracy and summary report from sklearn.

    Args:
        targets: target labels
        predictions: predicted labels
        output_dict: if True sklearn returns a summary report as dict, if False the
          report is in string format
        exclude_label: labels to exclude from evaluation

    Returns: a report from sklearn, precision, f1, and accuracy values
    """
    from sklearn import metrics

    targets = clean_labels(targets)
    predictions = clean_labels(predictions)

    labels = get_unique_labels(targets, exclude_label)
    if not labels:
        logger.warning("No labels to evaluate. Skip evaluation.")
        return {}, 0.0, 0.0, 0.0

    report = metrics.classification_report(
        targets, predictions, labels=labels, output_dict=output_dict
    )
    precision = metrics.precision_score(
        targets, predictions, labels=labels, average="weighted"
    )
    f1 = metrics.f1_score(targets, predictions, labels=labels, average="weighted")
    accuracy = metrics.accuracy_score(targets, predictions)

    return report, precision, f1, accuracy


def get_unique_labels(
    targets: Iterable[Text], exclude_label: Optional[Text]
) -> List[Text]:
    """Get unique labels. Exclude 'exclude_label' if specified.

    Args:
        targets: labels
        exclude_label: label to exclude

    Returns: unique list of labels
    """
    labels = set(targets)
    if exclude_label and exclude_label in labels:
        labels.remove(exclude_label)
    return list(labels)


def remove_empty_intent_examples(
    intent_results: List[IntentEvaluationResult],
) -> List[IntentEvaluationResult]:
    """Remove those examples without an intent.

    Args:
        intent_results: intent evaluation results

    Returns: intent evaluation results
    """
    filtered = []
    for r in intent_results:
        # substitute None values with empty string
        # to enable sklearn evaluation
        if r.intent_prediction is None:
            r = r._replace(intent_prediction="")

        if r.intent_target != "" and r.intent_target is not None:
            filtered.append(r)

    return filtered


def remove_empty_response_examples(
    response_results: List[ResponseSelectionEvaluationResult],
) -> List[ResponseSelectionEvaluationResult]:
    """Remove those examples without a response.

    Args:
        response_results: response selection evaluation results

    Returns: response selection evaluation results
    """

    filtered = []
    for r in response_results:
        # substitute None values with empty string
        # to enable sklearn evaluation
        if r.response_prediction is None:
            r = r._replace(response_prediction="")

        if r.response_target != "" and r.response_target is not None:
            filtered.append(r)

    return filtered


def clean_labels(labels: Iterable[Text]) -> List[Text]:
    """Remove `None` labels. sklearn metrics do not support them.

    Args:
        labels: list of labels

    Returns: cleaned labels
    """
    return [l if l is not None else "" for l in labels]


def drop_intents_below_freq(
    training_data: TrainingData, cutoff: int = 5
) -> TrainingData:
    """Remove intent groups with less than cutoff instances.

    Args:
        training_data: training data
        cutoff: threshold

    Returns: updated training data
    """
    logger.debug(
        "Raw data intent examples: {}".format(len(training_data.intent_examples))
    )
    keep_examples = [
        ex
        for ex in training_data.intent_examples
        if training_data.examples_per_intent[ex.get(INTENT)] >= cutoff
    ]

    return TrainingData(
        keep_examples, training_data.entity_synonyms, training_data.regex_features
    )


def write_intent_successes(
    intent_results: List[IntentEvaluationResult], successes_filename: Text
) -> None:
    """Write successful intent predictions to a file.

    Args:
        intent_results: intent evaluation result
        successes_filename: filename of file to save successful predictions to
    """
    successes = [
        {
            "text": r.message,
            "intent": r.intent_target,
            "intent_prediction": {
                "name": r.intent_prediction,
                "confidence": r.confidence,
            },
        }
        for r in intent_results
        if r.intent_target == r.intent_prediction
    ]

    if successes:
        io_utils.dump_obj_as_json_to_file(successes_filename, successes)
        logger.info(f"Successful intent predictions saved to {successes_filename}.")
        logger.debug(f"\n\nSuccessfully predicted the following intents: \n{successes}")
    else:
        logger.info("No successful intent predictions found.")


def write_intent_errors(
    intent_results: List[IntentEvaluationResult], errors_filename: Text
) -> None:
    """Write incorrect intent predictions to a file.

    Args:
        intent_results: intent evaluation result
        errors_filename: filename of file to save incorrect predictions to
    """
    errors = [
        {
            "text": r.message,
            "intent": r.intent_target,
            "intent_prediction": {
                "name": r.intent_prediction,
                "confidence": r.confidence,
            },
        }
        for r in intent_results
        if r.intent_target != r.intent_prediction
    ]

    if errors:
        io_utils.dump_obj_as_json_to_file(errors_filename, errors)
        logger.info(f"Incorrect intent predictions saved to {errors_filename}.")
        logger.debug(
            "\n\nThese intent examples could not be classified "
            "correctly: \n{}".format(errors)
        )
    else:
        logger.info("Your model predicted all intents successfully.")


def write_response_successes(
    response_results: List[ResponseSelectionEvaluationResult], successes_filename: Text
) -> None:
    """Write successful response selection predictions to a file.

    Args:
        response_results: response selection evaluation result
        successes_filename: filename of file to save successful predictions to
    """

    successes = [
        {
            "text": r.message,
            "intent_target": r.intent_target,
            "response_target": r.response_target,
            "response_prediction": {
                "name": r.response_prediction,
                "confidence": r.confidence,
            },
        }
        for r in response_results
        if r.response_prediction == r.response_target
    ]

    if successes:
        io_utils.dump_obj_as_json_to_file(successes_filename, successes)
        logger.info(f"Successful response predictions saved to {successes_filename}.")
        logger.debug(
            f"\n\nSuccessfully predicted the following responses: \n{successes}"
        )
    else:
        logger.info("No successful response predictions found.")


def write_response_errors(
    response_results: List[ResponseSelectionEvaluationResult], errors_filename: Text
) -> None:
    """Write incorrect response selection predictions to a file.

    Args:
        response_results: response selection evaluation result
        errors_filename: filename of file to save incorrect predictions to
    """
    errors = [
        {
            "text": r.message,
            "intent_target": r.intent_target,
            "response_target": r.response_target,
            "response_prediction": {
                "name": r.response_prediction,
                "confidence": r.confidence,
            },
        }
        for r in response_results
        if r.response_prediction != r.response_target
    ]

    if errors:
        io_utils.dump_obj_as_json_to_file(errors_filename, errors)
        logger.info(f"Incorrect response predictions saved to {errors_filename}.")
        logger.debug(
            "\n\nThese response examples could not be classified "
            "correctly: \n{}".format(errors)
        )
    else:
        logger.info("Your model predicted all responses successfully.")


def plot_attribute_confidences(
    results: Union[
        List[IntentEvaluationResult], List[ResponseSelectionEvaluationResult]
    ],
    hist_filename: Optional[Text],
    target_key: Text,
    prediction_key: Text,
    title: Text,
) -> None:
    """Create histogram of confidence distribution.

    Args:
        results: evaluation results
        hist_filename: filename to save plot to
        target_key: key of target in results
        prediction_key: key of predictions in results
        title: title of plot
    """
    pos_hist = [
        r.confidence
        for r in results
        if getattr(r, target_key) == getattr(r, prediction_key)
    ]

    neg_hist = [
        r.confidence
        for r in results
        if getattr(r, target_key) != getattr(r, prediction_key)
    ]

    plot_utils.plot_histogram([pos_hist, neg_hist], title, hist_filename)


def evaluate_response_selections(
    response_selection_results: List[ResponseSelectionEvaluationResult],
    output_directory: Optional[Text],
    successes: bool,
    errors: bool,
    disable_plotting: bool,
) -> Dict:  # pragma: no cover
    """Creates summary statistics for response selection.

    Only considers those examples with a set response.
    Others are filtered out. Returns a dictionary of containing the
    evaluation result.

    Args:
        response_selection_results: response selection evaluation results
        output_directory: directory to store files to
        successes: if True success are written down to disk
        errors: if True errors are written down to disk
        disable_plotting: if True no plots are created

    Returns: dictionary with evaluation results
    """
    import sklearn.metrics
    import sklearn.utils.multiclass

    # remove empty response targets
    num_examples = len(response_selection_results)
    response_selection_results = remove_empty_response_examples(
        response_selection_results
    )

    logger.info(
        f"Response Selection Evaluation: Only considering those "
        f"{len(response_selection_results)} examples that have a defined response out "
        f"of {num_examples} examples."
    )

    response_to_intent_target = {}
    for result in response_selection_results:
        response_to_intent_target[result.response_target] = result.intent_target

    target_responses, predicted_responses = _targets_predictions_from(
        response_selection_results, "response_target", "response_prediction"
    )

    confusion_matrix = sklearn.metrics.confusion_matrix(
        target_responses, predicted_responses
    )
    labels = sklearn.utils.multiclass.unique_labels(
        target_responses, predicted_responses
    )

    if output_directory:
        report, precision, f1, accuracy = get_evaluation_metrics(
            target_responses, predicted_responses, output_dict=True
        )
        report = _add_confused_labels_to_report(report, confusion_matrix, labels)

        report_filename = os.path.join(
            output_directory, "response_selection_report.json"
        )
        io_utils.dump_obj_as_json_to_file(report_filename, report)
        logger.info(f"Classification report saved to {report_filename}.")

    else:
        report, precision, f1, accuracy = get_evaluation_metrics(
            target_responses, predicted_responses
        )
        if isinstance(report, str):
            log_evaluation_table(report, precision, f1, accuracy)

    if successes:
        successes_filename = "response_selection_successes.json"
        if output_directory:
            successes_filename = os.path.join(output_directory, successes_filename)
        # save classified samples to file for debugging
        write_response_successes(response_selection_results, successes_filename)

    if errors:
        errors_filename = "response_selection_errors.json"
        if output_directory:
            errors_filename = os.path.join(output_directory, errors_filename)
        # log and save misclassified samples to file for debugging
        write_response_errors(response_selection_results, errors_filename)

    if not disable_plotting:
        confusion_matrix_filename = "response_selection_confusion_matrix.png"
        if output_directory:
            confusion_matrix_filename = os.path.join(
                output_directory, confusion_matrix_filename
            )
        _labels = [response_to_intent_target[label] for label in labels]
        plot_utils.plot_confusion_matrix(
            confusion_matrix,
            classes=_labels,
            title="Response Selection Confusion Matrix",
            output_file=confusion_matrix_filename,
        )

        histogram_filename = "response_selection_histogram.png"
        if output_directory:
            histogram_filename = os.path.join(output_directory, histogram_filename)
        plot_attribute_confidences(
            response_selection_results,
            histogram_filename,
            "response_target",
            "response_prediction",
            title="Response Selection Prediction Confidence Distribution",
        )

    predictions = [
        {
            "text": res.message,
            "intent_target": res.intent_target,
            "response_target": res.response_target,
            "response_predicted": res.response_prediction,
            "confidence": res.confidence,
        }
        for res in response_selection_results
    ]

    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def _add_confused_labels_to_report(
    report: Dict[Text, Dict[Text, Any]],
    confusion_matrix: np.ndarray,
    labels: List[Text],
) -> Dict[Text, Dict[Text, Union[Dict, Any]]]:
    """Adds a field "confused_with" to the evaluation report.

    The value is a dict of {"false_positive_label": false_positive_count} pairs.
    If there are no false positives in the confusion matrix,
    the dict will be empty. Typically we include the two most
    commonly false positive labels, three in the rare case that
    the diagonal element in the confusion matrix is not one of the
    three highest values in the row.

    Args:
        report: the evaluation report
        confusion_matrix: confusion matrix
        labels: list of labels

    Returns: updated evaluation report
    """
    # sort confusion matrix by false positives
    indices = np.argsort(confusion_matrix, axis=1)
    n_candidates = min(3, len(labels))

    for label in labels:
        # it is possible to predict intent 'None'
        if report.get(label):
            report[label]["confused_with"] = {}

    for i, label in enumerate(labels):
        for j in range(n_candidates):
            label_idx = indices[i, -(1 + j)]
            false_pos_label = labels[label_idx]
            false_positives = int(confusion_matrix[i, label_idx])
            if false_pos_label != label and false_positives > 0:
                report[label]["confused_with"][false_pos_label] = false_positives

    return report


def evaluate_intents(
    intent_results: List[IntentEvaluationResult],
    output_directory: Optional[Text],
    successes: bool,
    errors: bool,
    disable_plotting: bool,
) -> Dict:  # pragma: no cover
    """Creates summary statistics for intents.

    Only considers those examples with a set intent. Others are filtered out.
    Returns a dictionary of containing the evaluation result.

    Args:
        intent_results: intent evaluation results
        output_directory: directory to store files to
        successes: if True correct predictions are written to disk
        errors: if True incorrect predictions are written to disk
        disable_plotting: if True no plots are created

    Returns: dictionary with evaluation results
    """
    import sklearn.metrics
    import sklearn.utils.multiclass

    # remove empty intent targets
    num_examples = len(intent_results)
    intent_results = remove_empty_intent_examples(intent_results)

    logger.info(
        f"Intent Evaluation: Only considering those {len(intent_results)} examples "
        f"that have a defined intent out of {num_examples} examples."
    )

    target_intents, predicted_intents = _targets_predictions_from(
        intent_results, "intent_target", "intent_prediction"
    )

    confusion_matrix = sklearn.metrics.confusion_matrix(
        target_intents, predicted_intents
    )
    labels = sklearn.utils.multiclass.unique_labels(target_intents, predicted_intents)

    if output_directory:
        report, precision, f1, accuracy = get_evaluation_metrics(
            target_intents, predicted_intents, output_dict=True
        )
        report = _add_confused_labels_to_report(report, confusion_matrix, labels)

        report_filename = os.path.join(output_directory, "intent_report.json")
        io_utils.dump_obj_as_json_to_file(report_filename, report)
        logger.info(f"Classification report saved to {report_filename}.")

    else:
        report, precision, f1, accuracy = get_evaluation_metrics(
            target_intents, predicted_intents
        )
        if isinstance(report, str):
            log_evaluation_table(report, precision, f1, accuracy)

    if successes:
        successes_filename = "intent_successes.json"
        if output_directory:
            successes_filename = os.path.join(output_directory, successes_filename)
        # save classified samples to file for debugging
        write_intent_successes(intent_results, successes_filename)

    if errors:
        errors_filename = "intent_errors.json"
        if output_directory:
            errors_filename = os.path.join(output_directory, errors_filename)
        # log and save misclassified samples to file for debugging
        write_intent_errors(intent_results, errors_filename)

    if not disable_plotting:
        confusion_matrix_filename = "intent_confusion_matrix.png"
        if output_directory:
            confusion_matrix_filename = os.path.join(
                output_directory, confusion_matrix_filename
            )
        plot_utils.plot_confusion_matrix(
            confusion_matrix,
            classes=labels,
            title="Intent Confusion matrix",
            output_file=confusion_matrix_filename,
        )

        histogram_filename = "intent_histogram.png"
        if output_directory:
            histogram_filename = os.path.join(output_directory, histogram_filename)
        plot_attribute_confidences(
            intent_results,
            histogram_filename,
            "intent_target",
            "intent_prediction",
            title="Intent Prediction Confidence Distribution",
        )

    predictions = [
        {
            "text": res.message,
            "intent": res.intent_target,
            "predicted": res.intent_prediction,
            "confidence": res.confidence,
        }
        for res in intent_results
    ]

    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def merge_labels(
    aligned_predictions: List[Dict], extractor: Optional[Text] = None
) -> List[Text]:
    """Concatenates all labels of the aligned predictions.

    Takes the aligned prediction labels which are grouped for each message
    and concatenates them.

    Args:
        aligned_predictions: aligned predictions
        extractor: entity extractor name

    Returns: concatenated predictions
    """

    if extractor:
        label_lists = [ap["extractor_labels"][extractor] for ap in aligned_predictions]
    else:
        label_lists = [ap["target_labels"] for ap in aligned_predictions]

    return list(itertools.chain(*label_lists))


def substitute_labels(labels: List[Text], old: Text, new: Text) -> List[Text]:
    """Replaces label names in a list of labels.

    Args:
        labels: list of labels
        old: old label name that should be replaced
        new: new label name

    Returns: updated labels
    """
    return [new if label == old else label for label in labels]


def write_incorrect_entity_predictions(
    entity_results: List[EntityEvaluationResult],
    merged_targets: List[Text],
    merged_predictions: List[Text],
    error_filename: Text,
) -> None:
    """Write incorrect entity predictions to a file.

    Args:
        entity_results: response selection evaluation result
        merged_predictions: list of predicted entity labels
        merged_targets: list of true entity labels
        error_filename: filename of file to save incorrect predictions to
    """
    errors = collect_incorrect_entity_predictions(
        entity_results, merged_predictions, merged_targets
    )

    if errors:
        io_utils.dump_obj_as_json_to_file(error_filename, errors)
        logger.info(f"Incorrect entity predictions saved to {error_filename}.")
        logger.debug(
            "\n\nThese intent examples could not be classified "
            "correctly: \n{}".format(errors)
        )
    else:
        logger.info("Your model predicted all entities successfully.")


def collect_incorrect_entity_predictions(
    entity_results: List[EntityEvaluationResult],
    merged_predictions: List[Text],
    merged_targets: List[Text],
):
    """Get incorrect entity predictions.

    Args:
        entity_results: entity evaluation results
        merged_predictions: list of predicted entity labels
        merged_targets: list of true entity labels

    Returns: list of incorrect predictions
    """
    errors = []
    offset = 0
    for entity_result in entity_results:
        for i in range(offset, offset + len(entity_result.tokens)):
            if merged_targets[i] != merged_predictions[i]:
                errors.append(
                    {
                        "text": entity_result.message,
                        "entities": entity_result.entity_targets,
                        "predicted_entities": entity_result.entity_predictions,
                    }
                )
                break
        offset += len(entity_result.tokens)
    return errors


def write_successful_entity_predictions(
    entity_results: List[EntityEvaluationResult],
    merged_targets: List[Text],
    merged_predictions: List[Text],
    successes_filename: Text,
) -> None:
    """Write correct entity predictions to a file.

    Args:
        entity_results: response selection evaluation result
        merged_predictions: list of predicted entity labels
        merged_targets: list of true entity labels
        successes_filename: filename of file to save correct predictions to
    """
    successes = collect_successful_entity_predictions(
        entity_results, merged_predictions, merged_targets
    )

    if successes:
        io_utils.dump_obj_as_json_to_file(successes_filename, successes)
        logger.info(f"Successful entity predictions saved to {successes_filename}.")
        logger.debug(
            f"\n\nSuccessfully predicted the following entities: \n{successes}"
        )
    else:
        logger.info("No successful entity prediction found.")


def collect_successful_entity_predictions(
    entity_results: List[EntityEvaluationResult],
    merged_predictions: List[Text],
    merged_targets: List[Text],
):
    """Get correct entity predictions.

    Args:
        entity_results: entity evaluation results
        merged_predictions: list of predicted entity labels
        merged_targets: list of true entity labels

    Returns: list of correct predictions
    """
    successes = []
    offset = 0
    for entity_result in entity_results:
        for i in range(offset, offset + len(entity_result.tokens)):
            if (
                merged_targets[i] == merged_predictions[i]
                and merged_targets[i] != NO_ENTITY
            ):
                successes.append(
                    {
                        "text": entity_result.message,
                        "entities": entity_result.entity_targets,
                        "predicted_entities": entity_result.entity_predictions,
                    }
                )
                break
        offset += len(entity_result.tokens)
    return successes


def evaluate_entities(
    entity_results: List[EntityEvaluationResult],
    extractors: Set[Text],
    output_directory: Optional[Text],
    successes: bool = False,
    errors: bool = False,
) -> Dict:  # pragma: no cover
    """Creates summary statistics for each entity extractor.

    Logs precision, recall, and F1 per entity type for each extractor.

    Args:
        entity_results: entity evaluation results
        extractors: entity extractors to consider
        output_directory: directory to store files to
        successes: if True correct predictions are written to disk
        errors: if True incorrect predictions are written to disk

    Returns: dictionary with evaluation results
    """
    aligned_predictions = align_all_entity_predictions(entity_results, extractors)
    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, NO_ENTITY_TAG, NO_ENTITY)

    result = {}

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(
            merged_predictions, NO_ENTITY_TAG, NO_ENTITY
        )
        logger.info(f"Evaluation for entity extractor: {extractor} ")
        if output_directory:
            report_filename = f"{extractor}_report.json"
            extractor_report_filename = os.path.join(output_directory, report_filename)

            report, precision, f1, accuracy = get_evaluation_metrics(
                merged_targets,
                merged_predictions,
                output_dict=True,
                exclude_label=NO_ENTITY,
            )
            io_utils.dump_obj_as_json_to_file(extractor_report_filename, report)

            logger.info(
                "Classification report for '{}' saved to '{}'."
                "".format(extractor, extractor_report_filename)
            )

        else:
            report, precision, f1, accuracy = get_evaluation_metrics(
                merged_targets,
                merged_predictions,
                output_dict=False,
                exclude_label=NO_ENTITY,
            )
            if isinstance(report, str):
                log_evaluation_table(report, precision, f1, accuracy)

        if successes:
            successes_filename = f"{extractor}_successes.json"
            if output_directory:
                successes_filename = os.path.join(output_directory, successes_filename)
            # save classified samples to file for debugging
            write_successful_entity_predictions(
                entity_results, merged_targets, merged_predictions, successes_filename
            )

        if errors:
            errors_filename = f"{extractor}_errors.json"
            if output_directory:
                errors_filename = os.path.join(output_directory, errors_filename)
            # log and save misclassified samples to file for debugging
            write_incorrect_entity_predictions(
                entity_results, merged_targets, merged_predictions, errors_filename
            )

        result[extractor] = {
            "report": report,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy,
        }

    return result


def is_token_within_entity(token: Token, entity: Dict) -> bool:
    """Checks if a token is within the boundaries of an entity."""
    return determine_intersection(token, entity) == len(token.text)


def does_token_cross_borders(token: Token, entity: Dict) -> bool:
    """Checks if a token crosses the boundaries of an entity."""

    num_intersect = determine_intersection(token, entity)
    return 0 < num_intersect < len(token.text)


def determine_intersection(token: Token, entity: Dict) -> int:
    """Calculates how many characters a given token and entity share."""

    pos_token = set(range(token.start, token.end))
    pos_entity = set(range(entity["start"], entity["end"]))
    return len(pos_token.intersection(pos_entity))


def do_entities_overlap(entities: List[Dict]) -> bool:
    """Checks if entities overlap.

    I.e. cross each others start and end boundaries.

    Args:
        entities: list of entities

    Returns: true if entities overlap, false otherwise.
    """
    sorted_entities = sorted(entities, key=lambda e: e["start"])
    for i in range(len(sorted_entities) - 1):
        curr_ent = sorted_entities[i]
        next_ent = sorted_entities[i + 1]
        if (
            next_ent["start"] < curr_ent["end"]
            and next_ent["entity"] != curr_ent["entity"]
        ):
            logger.warning(f"Overlapping entity {curr_ent} with {next_ent}")
            return True

    return False


def find_intersecting_entities(token: Token, entities: List[Dict]) -> List[Dict]:
    """Finds the entities that intersect with a token.

    Args:
        token: a single token
        entities: entities found by a single extractor

    Returns: list of entities
    """
    candidates = []
    for e in entities:
        if is_token_within_entity(token, e):
            candidates.append(e)
        elif does_token_cross_borders(token, e):
            candidates.append(e)
            logger.debug(
                "Token boundary error for token {}({}, {}) "
                "and entity {}"
                "".format(token.text, token.start, token.end, e)
            )
    return candidates


def pick_best_entity_fit(
    token: Token,
    candidates: List[Dict[Text, Any]],
    attribute_key: Text = ENTITY_ATTRIBUTE_TYPE,
) -> Text:
    """
    Determines the token label for the provided attribute key given intersecting
    entities.

    Args:
        token: a single token
        candidates: entities found by a single extractor
        attribute_key: the attribute key of interest

    Returns:
        the value of the attribute key of the best fitting entity
    """
    if len(candidates) == 0:
        return NO_ENTITY_TAG
    elif len(candidates) == 1:
        return candidates[0].get(attribute_key) or NO_ENTITY_TAG
    else:
        best_fit = np.argmax([determine_intersection(token, c) for c in candidates])
        return candidates[best_fit].get(attribute_key) or NO_ENTITY_TAG


def determine_token_labels(
    token: Token,
    entities: List[Dict],
    extractors: Optional[Set[Text]] = None,
    attribute_key: Text = ENTITY_ATTRIBUTE_TYPE,
) -> Text:
    """
    Determines the token label for the provided attribute key given entities that do
    not overlap.

    Args:
        token: a single token
        entities: entities found by a single extractor
        extractors: list of extractors
        attribute_key: the attribute key for which the entity type should be returned
    Returns:
        entity type
    """

    if entities is None or len(entities) == 0:
        return NO_ENTITY_TAG
    if not do_extractors_support_overlap(extractors) and do_entities_overlap(entities):
        raise ValueError("The possible entities should not overlap")

    candidates = find_intersecting_entities(token, entities)
    return pick_best_entity_fit(token, candidates, attribute_key)


def do_extractors_support_overlap(extractors: Optional[Set[Text]]) -> bool:
    """Checks if extractors support overlapping entities"""
    if extractors is None:
        return False

    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

    return CRFEntityExtractor.name not in extractors


def align_entity_predictions(
    result: EntityEvaluationResult, extractors: Set[Text]
) -> Dict:
    """Aligns entity predictions to the message tokens.

    Determines for every token the true label based on the
    prediction targets and the label assigned by each
    single extractor.

    Args:
        result: entity evaluation result
        extractors: the entity extractors that should be considered

    Returns: dictionary containing the true token labels and token labels
             from the extractors
    """
    true_token_labels = []
    entities_by_extractors: Dict[Text, List] = {
        extractor: [] for extractor in extractors
    }
    for p in result.entity_predictions:
        entities_by_extractors[p[EXTRACTOR]].append(p)
    extractor_labels: Dict[Text, List] = {extractor: [] for extractor in extractors}
    for t in result.tokens:
        true_token_labels.append(_concat_entity_labels(t, result.entity_targets))
        for extractor, entities in entities_by_extractors.items():
            extracted = _concat_entity_labels(t, entities, {extractor})
            extractor_labels[extractor].append(extracted)

    return {
        "target_labels": true_token_labels,
        "extractor_labels": dict(extractor_labels),
    }


def _concat_entity_labels(
    token: Token, entities: List[Dict], extractors: Optional[Set[Text]] = None
) -> Text:
    """Concatenate labels for entity type, role, and group for evaluation.

    In order to calculate metrics also for entity type, role, and group we need to
    concatenate their labels. For example, 'location.destination'. This allows
    us to report metrics for every combination of entity type, role, and group.

    Args:
        token: the token we are looking at
        entities: the available entities
        extractors: the extractor of interest

    Returns:
        the entity label of the provided token
    """
    entity_label = determine_token_labels(
        token, entities, extractors, ENTITY_ATTRIBUTE_TYPE
    )
    group_label = determine_token_labels(
        token, entities, extractors, ENTITY_ATTRIBUTE_GROUP
    )
    role_label = determine_token_labels(
        token, entities, extractors, ENTITY_ATTRIBUTE_ROLE
    )

    if entity_label == role_label == group_label == NO_ENTITY_TAG:
        return NO_ENTITY_TAG

    labels = [entity_label, group_label, role_label]
    labels = [label for label in labels if label != NO_ENTITY_TAG]

    return ".".join(labels)


def align_all_entity_predictions(
    entity_results: List[EntityEvaluationResult], extractors: Set[Text]
) -> List[Dict]:
    """Aligns entity predictions to the message tokens for the whole dataset
    using align_entity_predictions.

    Args:
        entity_results: list of entity prediction results
        extractors: the entity extractors that should be considered

    Returns: list of dictionaries containing the true token labels and token
    labels from the extractors
    """
    aligned_predictions = []
    for result in entity_results:
        aligned_predictions.append(align_entity_predictions(result, extractors))

    return aligned_predictions


def get_eval_data(
    interpreter: Interpreter, test_data: TrainingData
) -> Tuple[
    List[IntentEvaluationResult],
    List[ResponseSelectionEvaluationResult],
    List[EntityEvaluationResult],
]:  # pragma: no cover
    """Runs the model for the test set and extracts targets and predictions.

    Returns intent results (intent targets and predictions, the original
    messages and the confidences of the predictions), response results (
    response targets and predictions) as well as entity results
    (entity_targets, entity_predictions, and tokens).

    Args:
        interpreter: the interpreter
        test_data: test data

    Returns: intent, response, and entity evaluation results
    """
    logger.info("Running model for predictions:")

    intent_results, entity_results, response_selection_results = [], [], []

    response_labels = [
        e.get("response")
        for e in test_data.intent_examples
        if e.get("response") is not None
    ]
    intent_labels = [e.get("intent") for e in test_data.intent_examples]
    should_eval_intents = (
        is_intent_classifier_present(interpreter) and len(set(intent_labels)) >= 2
    )
    should_eval_response_selection = (
        is_response_selector_present(interpreter) and len(set(response_labels)) >= 2
    )
    available_response_selector_types = get_available_response_selector_types(
        interpreter
    )

    should_eval_entities = is_entity_extractor_present(interpreter)

    for example in tqdm(test_data.training_examples):
        result = interpreter.parse(example.text, only_output_properties=False)

        if should_eval_intents:
            intent_prediction = result.get("intent", {}) or {}
            intent_results.append(
                IntentEvaluationResult(
                    example.get("intent", ""),
                    intent_prediction.get("name"),
                    result.get("text", {}),
                    intent_prediction.get("confidence"),
                )
            )

        if should_eval_response_selection:

            # including all examples here. Empty response examples are filtered at the
            # time of metric calculation
            intent_target = example.get("intent", "")
            selector_properties = result.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})

            if intent_target in available_response_selector_types:
                response_prediction_key = intent_target
            else:
                response_prediction_key = DEFAULT_OPEN_UTTERANCE_TYPE

            response_prediction = selector_properties.get(
                response_prediction_key, {}
            ).get(OPEN_UTTERANCE_PREDICTION_KEY, {})

            response_target = example.get("response", "")

            complete_intent = example.get_combined_intent_response_key()

            response_selection_results.append(
                ResponseSelectionEvaluationResult(
                    complete_intent,
                    response_target,
                    response_prediction.get("name"),
                    result.get("text", {}),
                    response_prediction.get("confidence"),
                )
            )

        if should_eval_entities:
            entity_results.append(
                EntityEvaluationResult(
                    example.get("entities", []),
                    result.get("entities", []),
                    result.get("tokens", []),
                    result.get("text", ""),
                )
            )

    return intent_results, response_selection_results, entity_results


def get_entity_extractors(interpreter: Interpreter) -> Set[Text]:
    """Finds the names of entity extractors used by the interpreter.

    Processors are removed since they do not detect the boundaries themselves.

    Args:
        interpreter: the interpreter

    Returns: entity extractor names
    """
    from rasa.nlu.extractors.extractor import EntityExtractor
    from rasa.nlu.classifiers.diet_classifier import DIETClassifier

    extractors = set()
    for c in interpreter.pipeline:
        if isinstance(c, EntityExtractor):
            if isinstance(c, DIETClassifier):
                if c.component_config[ENTITY_RECOGNITION]:
                    extractors.add(c.name)
            else:
                extractors.add(c.name)

    return extractors - ENTITY_PROCESSORS


def is_entity_extractor_present(interpreter: Interpreter) -> bool:
    """Checks whether entity extractor is present."""

    extractors = get_entity_extractors(interpreter)
    return extractors != []


def is_intent_classifier_present(interpreter: Interpreter) -> bool:
    """Checks whether intent classifier is present."""

    from rasa.nlu.classifiers.classifier import IntentClassifier

    intent_classifiers = [
        c.name for c in interpreter.pipeline if isinstance(c, IntentClassifier)
    ]
    return intent_classifiers != []


def is_response_selector_present(interpreter: Interpreter) -> bool:
    """Checks whether response selector is present."""

    from rasa.nlu.selectors.response_selector import ResponseSelector

    response_selectors = [
        c.name for c in interpreter.pipeline if isinstance(c, ResponseSelector)
    ]
    return response_selectors != []


def get_available_response_selector_types(interpreter: Interpreter) -> List[Text]:
    """Gets all available response selector types."""

    from rasa.nlu.selectors.response_selector import ResponseSelector

    response_selector_types = [
        c.retrieval_intent
        for c in interpreter.pipeline
        if isinstance(c, ResponseSelector)
    ]

    return response_selector_types


def remove_pretrained_extractors(pipeline: List[Component]) -> List[Component]:
    """Remove pre-trained extractors from the pipeline.

    Remove pre-trained extractors so that entities from pre-trained extractors
    are not predicted upon parsing.

    Args:
        pipeline: the pipeline

    Returns:
        Updated pipeline
    """
    pipeline = [c for c in pipeline if c.name not in PRETRAINED_EXTRACTORS]
    return pipeline


def run_evaluation(
    data_path: Text,
    model_path: Text,
    output_directory: Optional[Text] = None,
    successes: bool = False,
    errors: bool = False,
    component_builder: Optional[ComponentBuilder] = None,
    disable_plotting: bool = False,
) -> Dict:  # pragma: no cover
    """Evaluate intent classification, response selection and entity extraction.

    Args:
        data_path: path to the test data
        model_path: path to the model
        output_directory: path to folder where all output will be stored
        successes: if true successful predictions are written to a file
        errors: if true incorrect predictions are written to a file
        component_builder: component builder
        disable_plotting: if true confusion matrix and histogram will not be rendered

    Returns: dictionary containing evaluation results
    """
    import rasa.nlu.training_data

    # get the metadata config from the package data
    interpreter = Interpreter.load(model_path, component_builder)

    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)
    test_data = rasa.nlu.training_data.load_data(
        data_path, interpreter.model_metadata.language
    )

    result: Dict[Text, Optional[Dict]] = {
        "intent_evaluation": None,
        "entity_evaluation": None,
        "response_selection_evaluation": None,
    }

    if output_directory:
        io_utils.create_directory(output_directory)

    intent_results, response_selection_results, entity_results, = get_eval_data(
        interpreter, test_data
    )

    if intent_results:
        logger.info("Intent evaluation results:")
        result["intent_evaluation"] = evaluate_intents(
            intent_results, output_directory, successes, errors, disable_plotting
        )

    if response_selection_results:
        logger.info("Response selection evaluation results:")
        result["response_selection_evaluation"] = evaluate_response_selections(
            response_selection_results,
            output_directory,
            successes,
            errors,
            disable_plotting,
        )

    if entity_results:
        logger.info("Entity evaluation results:")
        extractors = get_entity_extractors(interpreter)
        result["entity_evaluation"] = evaluate_entities(
            entity_results, extractors, output_directory, successes, errors
        )

    return result


def generate_folds(
    n: int, training_data: TrainingData
) -> Iterator[Tuple[TrainingData, TrainingData]]:
    """Generates n cross validation folds for given training data."""

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n, shuffle=True)
    x = training_data.intent_examples

    # Get labels with response key appended to intent name because we want a
    # stratified split on all intents(including retrieval intents if they exist)
    y = [example.get_combined_intent_response_key() for example in x]
    for i_fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        logger.debug(f"Fold: {i_fold}")
        train = [x[i] for i in train_index]
        test = [x[i] for i in test_index]
        yield (
            TrainingData(
                training_examples=train,
                entity_synonyms=training_data.entity_synonyms,
                regex_features=training_data.regex_features,
            ),
            TrainingData(
                training_examples=test,
                entity_synonyms=training_data.entity_synonyms,
                regex_features=training_data.regex_features,
            ),
        )


def combine_result(
    intent_metrics: IntentMetrics,
    entity_metrics: EntityMetrics,
    response_selection_metrics: ResponseSelectionMetrics,
    interpreter: Interpreter,
    data: TrainingData,
    intent_results: Optional[List[IntentEvaluationResult]] = None,
    entity_results: Optional[List[EntityEvaluationResult]] = None,
    response_selection_results: Optional[
        List[ResponseSelectionEvaluationResult]
    ] = None,
) -> Tuple[IntentMetrics, EntityMetrics, ResponseSelectionMetrics]:
    """Collects intent, response selection and entity metrics for cross validation
    folds.

    If `intent_results`, `response_selection_results` or `entity_results` is provided
    as a list, prediction results are also collected.

    Args:
        intent_metrics: intent metrics
        entity_metrics: entity metrics
        response_selection_metrics: response selection metrics
        interpreter: the interpreter
        data: training data
        intent_results: intent evaluation results
        entity_results: entity evaluation results
        response_selection_results: reponse selection evaluation results

    Returns: intent, entity, and response selection metrics
    """
    (
        intent_current_metrics,
        entity_current_metrics,
        response_selection_current_metrics,
        current_intent_results,
        current_entity_results,
        current_response_selection_results,
    ) = compute_metrics(interpreter, data)

    if intent_results is not None:
        intent_results += current_intent_results

    if entity_results is not None:
        entity_results += current_entity_results

    if response_selection_results is not None:
        response_selection_results += current_response_selection_results

    for k, v in intent_current_metrics.items():
        intent_metrics[k] = v + intent_metrics[k]

    for k, v in response_selection_current_metrics.items():
        response_selection_metrics[k] = v + response_selection_metrics[k]

    for extractor, extractor_metric in entity_current_metrics.items():
        entity_metrics[extractor] = {
            k: v + entity_metrics[extractor][k] for k, v in extractor_metric.items()
        }

    return intent_metrics, entity_metrics, response_selection_metrics


def _contains_entity_labels(entity_results: List[EntityEvaluationResult]) -> bool:

    for result in entity_results:
        if result.entity_targets or result.entity_predictions:
            return True


def cross_validate(
    data: TrainingData,
    n_folds: int,
    nlu_config: Union[RasaNLUModelConfig, Text],
    output: Optional[Text] = None,
    successes: bool = False,
    errors: bool = False,
    disable_plotting: bool = False,
) -> Tuple[CVEvaluationResult, CVEvaluationResult, CVEvaluationResult]:
    """Stratified cross validation on data.

    Args:
        data: Training Data
        n_folds: integer, number of cv folds
        nlu_config: nlu config file
        output: path to folder where reports are stored
        successes: if true successful predictions are written to a file
        errors: if true incorrect predictions are written to a file
        disable_plotting: if true no confusion matrix and historgram plates are created

    Returns:
        dictionary with key, list structure, where each entry in list
              corresponds to the relevant result for one fold
    """
    import rasa.nlu.config
    from collections import defaultdict

    if isinstance(nlu_config, str):
        nlu_config = rasa.nlu.config.load(nlu_config)

    if output:
        io_utils.create_directory(output)

    trainer = Trainer(nlu_config)
    trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)

    intent_train_metrics: IntentMetrics = defaultdict(list)
    intent_test_metrics: IntentMetrics = defaultdict(list)
    entity_train_metrics: EntityMetrics = defaultdict(lambda: defaultdict(list))
    entity_test_metrics: EntityMetrics = defaultdict(lambda: defaultdict(list))
    response_selection_train_metrics: ResponseSelectionMetrics = defaultdict(list)
    response_selection_test_metrics: ResponseSelectionMetrics = defaultdict(list)

    intent_test_results: List[IntentEvaluationResult] = []
    entity_test_results: List[EntityEvaluationResult] = []
    response_selection_test_results: List[ResponseSelectionEvaluationResult] = ([])
    intent_classifier_present = False
    response_selector_present = False
    entity_evaluation_possible = False
    extractors: Set[Text] = set()

    for train, test in generate_folds(n_folds, data):
        interpreter = trainer.train(train)

        # calculate train accuracy
        combine_result(
            intent_train_metrics,
            entity_train_metrics,
            response_selection_train_metrics,
            interpreter,
            train,
        )
        # calculate test accuracy
        combine_result(
            intent_test_metrics,
            entity_test_metrics,
            response_selection_test_metrics,
            interpreter,
            test,
            intent_test_results,
            entity_test_results,
            response_selection_test_results,
        )

        if not extractors:
            extractors = get_entity_extractors(interpreter)
            entity_evaluation_possible = (
                entity_evaluation_possible
                or _contains_entity_labels(entity_test_results)
            )

        if is_intent_classifier_present(interpreter):
            intent_classifier_present = True

        if is_response_selector_present(interpreter):
            response_selector_present = True

    if intent_classifier_present and intent_test_results:
        logger.info("Accumulated test folds intent evaluation results:")
        evaluate_intents(
            intent_test_results, output, successes, errors, disable_plotting
        )

    if extractors and entity_evaluation_possible:
        logger.info("Accumulated test folds entity evaluation results:")
        evaluate_entities(entity_test_results, extractors, output, successes, errors)

    if response_selector_present and response_selection_test_results:
        logger.info("Accumulated test folds response selection evaluation results:")
        evaluate_response_selections(
            response_selection_test_results, output, successes, errors, disable_plotting
        )

    if not entity_evaluation_possible:
        entity_test_metrics = defaultdict(lambda: defaultdict(list))
        entity_train_metrics = defaultdict(lambda: defaultdict(list))

    return (
        CVEvaluationResult(dict(intent_train_metrics), dict(intent_test_metrics)),
        CVEvaluationResult(dict(entity_train_metrics), dict(entity_test_metrics)),
        CVEvaluationResult(
            dict(response_selection_train_metrics),
            dict(response_selection_test_metrics),
        ),
    )


def _targets_predictions_from(
    results: Union[
        List[IntentEvaluationResult], List[ResponseSelectionEvaluationResult]
    ],
    target_key: Text,
    prediction_key: Text,
) -> Iterator[Iterable[Optional[Text]]]:
    return zip(*[(getattr(r, target_key), getattr(r, prediction_key)) for r in results])


def compute_metrics(
    interpreter: Interpreter, training_data: TrainingData
) -> Tuple[
    IntentMetrics,
    EntityMetrics,
    ResponseSelectionMetrics,
    List[IntentEvaluationResult],
    List[EntityEvaluationResult],
    List[ResponseSelectionEvaluationResult],
]:
    """Computes metrics for intent classification, response selection and entity
    extraction.

    Args:
        interpreter: the interpreter
        training_data: training data

    Returns: intent, response selection and entity metrics, and prediction results.
    """
    intent_results, response_selection_results, entity_results = get_eval_data(
        interpreter, training_data
    )

    intent_results = remove_empty_intent_examples(intent_results)

    response_selection_results = remove_empty_response_examples(
        response_selection_results
    )

    intent_metrics = {}
    if intent_results:
        intent_metrics = _compute_metrics(
            intent_results, "intent_target", "intent_prediction"
        )

    entity_metrics = {}
    if entity_results:
        entity_metrics = _compute_entity_metrics(entity_results, interpreter)

    response_selection_metrics = {}
    if response_selection_results:
        response_selection_metrics = _compute_metrics(
            response_selection_results, "response_target", "response_prediction"
        )

    return (
        intent_metrics,
        entity_metrics,
        response_selection_metrics,
        intent_results,
        entity_results,
        response_selection_results,
    )


def compare_nlu(
    configs: List[Text],
    data: TrainingData,
    exclusion_percentages: List[int],
    f_score_results: Dict[Text, Any],
    model_names: List[Text],
    output: Text,
    runs: int,
) -> List[int]:
    """
    Trains and compares multiple NLU models.
    For each run and exclusion percentage a model per config file is trained.
    Thereby, the model is trained only on the current percentage of training data.
    Afterwards, the model is tested on the complete test data of that run.
    All results are stored in the provided output directory.

    Args:
        configs: config files needed for training
        data: training data
        exclusion_percentages: percentages of training data to exclude during comparison
        f_score_results: dictionary of model name to f-score results per run
        model_names: names of the models to train
        output: the output directory
        runs: number of comparison runs

    Returns: training examples per run
    """

    from rasa.train import train_nlu

    training_examples_per_run = []

    for run in range(runs):

        logger.info("Beginning comparison run {}/{}".format(run + 1, runs))

        run_path = os.path.join(output, "run_{}".format(run + 1))
        io_utils.create_path(run_path)

        test_path = os.path.join(run_path, TEST_DATA_FILE)
        io_utils.create_path(test_path)

        train, test = data.train_test_split()
        io_utils.write_text_file(test.nlu_as_markdown(), test_path)

        for percentage in exclusion_percentages:
            percent_string = f"{percentage}%_exclusion"

            _, train = train.train_test_split(percentage / 100)
            # only count for the first run and ignore the others
            if run == 0:
                training_examples_per_run.append(len(train.training_examples))

            model_output_path = os.path.join(run_path, percent_string)
            train_split_path = os.path.join(model_output_path, "train")
            train_nlu_split_path = os.path.join(train_split_path, TRAIN_DATA_FILE)
            train_nlg_split_path = os.path.join(train_split_path, NLG_DATA_FILE)
            io_utils.create_path(train_nlu_split_path)
            io_utils.write_text_file(train.nlu_as_markdown(), train_nlu_split_path)
            io_utils.write_text_file(train.nlg_as_markdown(), train_nlg_split_path)

            for nlu_config, model_name in zip(configs, model_names):
                logger.info(
                    "Evaluating configuration '{}' with {} training data.".format(
                        model_name, percent_string
                    )
                )

                try:
                    model_path = train_nlu(
                        nlu_config,
                        train_split_path,
                        model_output_path,
                        fixed_model_name=model_name,
                    )
                except Exception as e:
                    logger.warning(f"Training model '{model_name}' failed. Error: {e}")
                    f_score_results[model_name][run].append(0.0)
                    continue

                model_path = os.path.join(get_model(model_path), "nlu")

                output_path = os.path.join(model_output_path, f"{model_name}_report")
                result = run_evaluation(
                    test_path, model_path, output_directory=output_path, errors=True
                )

                f1 = result["intent_evaluation"]["f1_score"]
                f_score_results[model_name][run].append(f1)

    return training_examples_per_run


def _compute_metrics(
    results: Union[
        List[IntentEvaluationResult], List[ResponseSelectionEvaluationResult]
    ],
    target_key: Text,
    prediction_key: Text,
) -> Union[IntentMetrics, ResponseSelectionMetrics]:
    """Computes evaluation metrics for a given corpus and returns the results.

    Args:
        results: evaluation results
        target_key: target key name
        prediction_key: prediction key name

    Returns: metrics
    """
    # compute fold metrics
    targets, predictions = _targets_predictions_from(
        results, target_key, prediction_key
    )
    _, precision, f1, accuracy = get_evaluation_metrics(targets, predictions)

    return {"Accuracy": [accuracy], "F1-score": [f1], "Precision": [precision]}


def _compute_entity_metrics(
    entity_results: List[EntityEvaluationResult], interpreter: Interpreter
) -> EntityMetrics:
    """Computes entity evaluation metrics and returns the results.

    Args:
        entity_results: entity evaluation results
        interpreter: the interpreter

    Returns: entity metrics
    """
    entity_metric_results: EntityMetrics = defaultdict(lambda: defaultdict(list))
    extractors = get_entity_extractors(interpreter)

    if not extractors:
        return entity_metric_results

    aligned_predictions = align_all_entity_predictions(entity_results, extractors)

    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, NO_ENTITY_TAG, NO_ENTITY)

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(
            merged_predictions, NO_ENTITY_TAG, NO_ENTITY
        )
        _, precision, f1, accuracy = get_evaluation_metrics(
            merged_targets, merged_predictions, exclude_label=NO_ENTITY
        )
        entity_metric_results[extractor]["Accuracy"].append(accuracy)
        entity_metric_results[extractor]["F1-score"].append(f1)
        entity_metric_results[extractor]["Precision"].append(precision)

    return entity_metric_results


def log_results(results: IntentMetrics, dataset_name: Text) -> None:
    """Logs results of cross validation.

    Args:
        results: dictionary of results returned from cross validation
        dataset_name: string of which dataset the results are from, e.g. test/train
    """
    for k, v in results.items():
        logger.info(
            "{} {}: {:.3f} ({:.3f})".format(dataset_name, k, np.mean(v), np.std(v))
        )


def log_entity_results(results: EntityMetrics, dataset_name: Text) -> None:
    """Logs entity results of cross validation.

    Args:
        results: dictionary of dictionaries of results returned from cross validation
        dataset_name: string of which dataset the results are from, e.g. test/train
    """
    for extractor, result in results.items():
        logger.info(f"Entity extractor: {extractor}")
        log_results(result, dataset_name)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.test` directly is no longer supported. Please use "
        "`rasa test` to test a combined Core and NLU model or `rasa test nlu` "
        "to test an NLU model."
    )
