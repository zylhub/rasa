:desc: Set up a pipeline of components.

.. _choosing-a-pipeline:

Choosing a Pipeline
===================

.. edit-link::

Choosing an NLU pipeline allows you to customize your model and finetune
it on your dataset.

.. contents::
   :local:

.. note::
    With Rasa 1.8.0 we updated some components and deprecated all existing pipeline templates.
    However, **any of the old terminology will still behave the same way as it did before**!

.. warning::
    We deprecated all existing pipeline templates, e.g.
    :ref:`supervised_embeddings <section_supervised_embeddings_pipeline>`,
    :ref:`pretrained_embeddings_spacy <section_pretrained_embeddings_spacy_pipeline>` and
    :ref:`pretrained_embeddings_convert <section_pretrained_embeddings_convert_pipeline>`. Please list any
    components you want to use directly in the configuration file.

The Short Answer
----------------

If your training data is in English, a good starting point is the following pipeline:

.. literalinclude:: ../../data/configs_for_docs/default_english_config.yml
    :language: yaml

In case your training data is in a different language than English, use the following pipeline:

.. literalinclude:: ../../data/configs_for_docs/default_config.yml
    :language: yaml


A Longer Answer
---------------

We recommend using following pipeline, if your training data is in English:

.. literalinclude:: ../../data/configs_for_docs/default_english_config.yml
    :language: yaml

The pipeline contains the :ref:`ConveRTFeaturizer` that provides pre-trained word embeddings of the user utterance.
Pre-trained word embeddings are helpful as they already encode some kind of linguistic knowledge.
For example, if you have a sentence like "I want to buy apples" in your training data, and Rasa is asked to predict
the intent for "get pears", your model already knows that the words "apples" and "pears" are very similar.
This is especially useful if you don’t have enough training data.
The advantage of the :ref:`ConveRTFeaturizer` is that it doesn't treat each word of the user message independently, but
creates a contextual vector representation for the complete sentence.
However, ``ConveRT`` is only available in English.

If your training data is not in English, but you still want to use pre-trained word embeddings, we recommend using
the following pipeline:

.. literalinclude:: ../../data/configs_for_docs/default_spacy_config.yml
    :language: yaml

It uses the :ref:`SpacyFeaturizer` instead of the :ref:`ConveRTFeaturizer`.
:ref:`SpacyFeaturizer` provides pre-trained word embeddings from either GloVe or fastText in many different languages
(see :ref:`pretrained-word-vectors`).

If you don't use any pre-trained word embeddings inside your pipeline, you are not bound to a specific language
and can train your model to be more domain specific.
If there are no word embeddings for your language or you have very domain specific terminology,
we recommend using the following pipeline:

.. literalinclude:: ../../data/configs_for_docs/default_config.yml
    :language: yaml

.. note::
    We encourage everyone to define their own pipeline by listing the names of the components you want to use.
    You can find the details of each component in :ref:`components`.
    If you want to use custom components in your pipeline, see :ref:`custom-nlu-components`.

Choosing the right Components
-----------------------------

A pipeline usually consist of three main parts:

    1. Tokenizaion
    2. Featurization
    3. Entity Recognition / Intent Classification / Response Selectors

Tokenization
~~~~~~~~~~~~
If your chosen language is whitespace-tokenized (words are separated by spaces), you
can use the :ref:`WhitespaceTokenizer`. If this is not the case you should use a different tokenizer.
We support a number of different :ref:`tokenizers <tokenizers>`, or you can
create your own :ref:`custom tokenizer <custom-nlu-components>`.

.. note::
    Some components further down the pipeline may require a specific tokenizer. You can find those requirements
    on the individual components in :ref:`components`. If a required component is missing inside the pipeline, an
    error will be thrown.

Featurization
~~~~~~~~~~~~~
You need to decide whether to use components that provide pre-trained word embeddings or not.

If you don't use any pre-trained word embeddings inside your pipeline, you are not bound to a specific language
and can train your model to be more domain specific. For example, in general English, the word "balance" is closely
related to "symmetry", but very different to the word "cash". In a banking domain, "balance" and "cash" are closely
related and you'd like your model to capture that.
You should only use featurizers from the category :ref:`sparse featurizers <text-featurizers>`, such as
:ref:`CountVectorsFeaturizer`, :ref:`RegexFeaturizer` or :ref:`LexicalSyntacticFeaturizer`, if you don't want to use
pre-trained word embeddings.

The advantage of using pre-trained word embeddings in your pipeline is that if you have a training example like:
"I want to buy apples", and Rasa is asked to predict the intent for "get pears", your model already knows that the
words "apples" and "pears" are very similar. This is especially useful if you don't have enough training data.
We support a few components that provide pre-trained word embeddings:

1. :ref:`MitieFeaturizer`
2. :ref:`SpacyFeaturizer`
3. :ref:`ConveRTFeaturizer`
4. :ref:`LanguageModelFeaturizer`

If your training data is in English, we recommend using the :ref:`ConveRTFeaturizer`.
The advantage of the :ref:`ConveRTFeaturizer` is that it doesn't treat each word of the user message independently, but
creates a contextual vector representation for the complete sentence. For example, if you
have a training example, like: "Can I book a car?", and Rasa is asked to predict the intent for "I need a ride from
my place", since the contextual vector representation for both examples are already very similar, the intent classified
for both is highly likely to be the same. This is also useful if you don't have enough training data.

An alternative to :ref:`ConveRTFeaturizer` is the :ref:`LanguageModelFeaturizer` which uses pre-trained language
models such as BERT, GPT-2, etc. to extract similar contextual vector representations for the complete sentence. See
:ref:`HFTransformersNLP` for a full list of supported language models.

If your training data is not in English you can also use a different variant of a language model which
is pre-trained in the language specific to your training data.
For example, there are chinese (``bert-base-chinese``) and japanese (``bert-base-japanese``) variants of the BERT model.
A full list of different variants of
these language models is available in the
`official documentation of the Transformers library <https://huggingface.co/transformers/pretrained_models.html>`_.

:ref:`SpacyFeaturizer` also provides word embeddings in many different languages (see :ref:`pretrained-word-vectors`),
so you can use this as another alternative, depending on the language of your training data.

Entity Recognition / Intent Classification / Response Selectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Depending on your data you may want to only perform intent classification, entity recognition or response selection.
Or you might want to combine multiple of those tasks.
We support several components for each of the tasks. All of them are listed in :ref:`components`.
We recommend using :ref:`diet-classifier` for intent classification and entity recognition
and :ref:`response-selector` for response selection.

Comparing different pipelines for your data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rasa gives you the tools to compare the performance of both of these pipelines on your data directly,
see :ref:`comparing-nlu-pipelines`.

.. note::

    Intent classification is independent of entity extraction. So sometimes
    NLU will get the intent right but entities wrong, or the other way around.
    You need to provide enough data for both intents and entities.


Class imbalance
---------------

Classification algorithms often do not perform well if there is a large `class imbalance`,
for example if you have a lot of training data for some intents and very little training data for others.
To mitigate this problem, you can use a ``balanced`` batching strategy.
This algorithm ensures that all classes are represented in every batch, or at least in
as many subsequent batches as possible, still mimicking the fact that some classes are more frequent than others.
Balanced batching is used by default. In order to turn it off and use a classic batching strategy include
``batch_strategy: sequence`` in your config file.

.. code-block:: yaml

    language: "en"

    pipeline:
    # - ... other components
    - name: "DIETClassifier"
      batch_strategy: sequence


Multiple Intents
----------------

If you want to split intents into multiple labels, e.g. for predicting multiple intents or for modeling hierarchical
intent structure, you need to use the :ref:`diet-classifier` in your pipeline.
You'll also need to define these flags in whichever tokenizer you are using:

    - ``intent_tokenization_flag``: Set it to ``True``, so that intent labels are tokenized.
    - ``intent_split_symbol``: Set it to the delimiter string that splits the intent labels. Default ``_``.

Read a `tutorial <https://blog.rasa.com/how-to-handle-multiple-intents-per-input-using-rasa-nlu-tensorflow-pipeline/>`__
on how to use multiple intents in Rasa.

Here's an example configuration:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
      intent_tokenization_flag: True
      intent_split_symbol: "_"
    - name: "CountVectorsFeaturizer"
    - name: "DIETClassifier"


Understanding the Rasa NLU Pipeline
-----------------------------------

In Rasa NLU, incoming messages are processed by a sequence of components.
These components are executed one after another in a so-called processing pipeline.
There are components for entity extraction, for intent classification, response selection,
pre-processing, and others. If you want to add your own component, for example to run a spell-check or to
do sentiment analysis, check out :ref:`custom-nlu-components`.

Each component processes the input and creates an output. The output can be used by any component that comes after
this component in the pipeline. There are components which only produce information that is used by other components
in the pipeline and there are other components that produce ``output`` attributes which will be returned after
the processing has finished. For example, for the sentence ``"I am looking for Chinese food"`` the output is:

.. code-block:: json

    {
        "text": "I am looking for Chinese food",
        "entities": [
            {
                "start": 8,
                "end": 15,
                "value": "chinese",
                "entity": "cuisine",
                "extractor": "DIETClassifier",
                "confidence": 0.864
            }
        ],
        "intent": {"confidence": 0.6485910906220309, "name": "restaurant_search"},
        "intent_ranking": [
            {"confidence": 0.6485910906220309, "name": "restaurant_search"},
            {"confidence": 0.1416153159565678, "name": "affirm"}
        ]
    }

This is created as a combination of the results of the different components in the following pipeline:

.. literalinclude:: ../../data/configs_for_docs/default_config.yml
    :language: yaml

For example, the ``entities`` attribute is created by the ``DIETClassifier`` component.


.. _section_component_lifecycle:

Component Lifecycle
-------------------

Every component can implement several methods from the ``Component`` base class; in a pipeline these different methods
will be called in a specific order. Lets assume, we added the following pipeline to our config:
``"pipeline": ["Component A", "Component B", "Last Component"]``.
The image shows the call order during the training of this pipeline:

.. image:: /_static/images/component_lifecycle.png

Before the first component is created using the ``create`` function, a so
called ``context`` is created (which is nothing more than a python dict).
This context is used to pass information between the components. For example,
one component can calculate feature vectors for the training data, store
that within the context and another component can retrieve these feature
vectors from the context and do intent classification.

Initially the context is filled with all configuration values, the arrows
in the image show the call order and visualize the path of the passed
context. After all components are trained and persisted, the
final context dictionary is used to persist the model's metadata.


The "entity" object explained
-----------------------------
After parsing, the entity is returned as a dictionary.  There are two fields that show information
about how the pipeline impacted the entities returned: the ``extractor`` field
of an entity tells you which entity extractor found this particular entity, and
the ``processors`` field contains the name of components that altered this
specific entity.

The use of synonyms can also cause the ``value`` field not match the ``text``
exactly. Instead it will return the trained synonym.

.. code-block:: json

    {
      "text": "show me chinese restaurants",
      "intent": "restaurant_search",
      "entities": [
        {
          "start": 8,
          "end": 15,
          "value": "chinese",
          "entity": "cuisine",
          "extractor": "CRFEntityExtractor",
          "confidence": 0.854,
          "processors": []
        }
      ]
    }

.. note::

    The ``confidence`` will be set by the ``CRFEntityExtractor`` and ``DIETClassifier`` component. The
    ``DucklingHTTPExtractor`` will always return ``1``. The ``SpacyEntityExtractor`` extractor does not provide this
    information and returns ``null``.



Pipeline Templates (deprecated)
-------------------------------

A template is just a shortcut for a full list of components. For example, these two configurations are equivalent:

.. code-block:: yaml

    language: "en"

    pipeline: "pretrained_embeddings_spacy"

and

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "SpacyNLP"
    - name: "SpacyTokenizer"
    - name: "SpacyFeaturizer"
    - name: "RegexFeaturizer"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"
    - name: "SklearnIntentClassifier"

The three most important pipelines are ``supervised_embeddings``, ``pretrained_embeddings_convert`` and
``pretrained_embeddings_spacy``.
The ``pretrained_embeddings_spacy`` pipeline uses pre-trained word vectors from either GloVe or fastText,
whereas ``pretrained_embeddings_convert`` uses a pretrained sentence encoding model
`ConveRT <https://github.com/PolyAI-LDN/polyai-models>`_ to extract vector representations of complete user
utterance as a whole. On the other hand, the ``supervised_embeddings`` pipeline doesn't use any pre-trained word
vectors or sentence vectors, but instead fits these specifically for your dataset.

.. note::
    These recommendations are highly dependent on your dataset and hence approximate. We suggest experimenting with
    different pipelines to train the best model.

.. _section_pretrained_embeddings_spacy_pipeline:

pretrained_embeddings_spacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The advantage of ``pretrained_embeddings_spacy`` pipeline is that if you have a training example like:
"I want to buy apples", and Rasa is asked to predict the intent for "get pears", your model
already knows that the words "apples" and "pears" are very similar. This is especially useful
if you don't have enough training data.

To use the ``pretrained_embeddings_spacy`` template, use the following configuration:

.. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_spacy_config_1.yml
    :language: yaml

See :ref:`pretrained-word-vectors` for more information about loading spacy language models.
To use the components and configure them separately:

.. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_spacy_config_2.yml
    :language: yaml

.. _section_pretrained_embeddings_convert_pipeline:

pretrained_embeddings_convert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. note::
        Since ``ConveRT`` model is trained only on an **English** corpus of conversations, this pipeline should only
        be used if your training data is in English language.

This pipeline uses the `ConveRT <https://github.com/PolyAI-LDN/polyai-models>`_ model to extract a vector representation of
a sentence and feeds them to the ``EmbeddingIntentClassifier`` for intent classification.
The advantage of using the ``pretrained_embeddings_convert`` pipeline is that it doesn't treat each word of the user
message independently, but creates a contextual vector representation for the complete sentence. For example, if you
have a training example, like: "can I book a car?", and Rasa is asked to predict the intent for "I need a ride from
my place", since the contextual vector representation for both examples are already very similar, the intent classified
for both is highly likely to be the same. This is also useful if you don't have enough training data.

    .. note::
        To use ``pretrained_embeddings_convert`` pipeline, you should install Rasa with ``pip install rasa[convert]``.
        Please also note that one of the dependencies(``tensorflow-text``) is currently only supported on Linux
        platforms.

To use the ``pretrained_embeddings_convert`` template:

.. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_convert_config_2.yml
    :language: yaml

To use the components and configure them separately:

.. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_convert_config_2.yml
    :language: yaml

.. _section_supervised_embeddings_pipeline:

supervised_embeddings
~~~~~~~~~~~~~~~~~~~~~

The advantage of the ``supervised_embeddings`` pipeline is that your word vectors will be customised
for your domain. For example, in general English, the word "balance" is closely related to "symmetry",
but very different to the word "cash". In a banking domain, "balance" and "cash" are closely related
and you'd like your model to capture that. This pipeline doesn't use a language-specific model,
so it will work with any language that you can tokenize (on whitespace or using a custom tokenizer).

You can read more about this topic `in this blog post <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`__ .

To train a Rasa model in your preferred language, define the
``supervised_embeddings`` pipeline as your pipeline in your ``config.yml`` or other configuration file:

.. literalinclude:: ../../data/configs_for_docs/supervised_embeddings_config_1.yml
    :language: yaml

The ``supervised_embeddings`` pipeline supports any language that can be whitespace tokenized. By default it uses
whitespace for tokenization. You can customize the setup of this pipeline by adding or changing components. Here are
the default components that make up the ``supervised_embeddings`` pipeline:

.. literalinclude:: ../../data/configs_for_docs/supervised_embeddings_config_2.yml
    :language: yaml

So for example, if your chosen language is not whitespace-tokenized (words are not separated by spaces), you
can replace the ``WhitespaceTokenizer`` with your own tokenizer. We support a number of different :ref:`tokenizers <tokenizers>`,
or you can :ref:`create your own <custom-nlu-components>`.

The pipeline uses two instances of ``CountVectorsFeaturizer``. The first one
featurizes text based on words. The second one featurizes text based on character
n-grams, preserving word boundaries. We empirically found the second featurizer
to be more powerful, but we decided to keep the first featurizer as well to make
featurization more robust.

.. _section_mitie_pipeline:

MITIE
~~~~~

You can also use MITIE as a source of word vectors in your pipeline.
The MITIE backend performs well for small datasets, but training can take very long if you have more than a couple
of hundred examples.

However, we do not recommend that you use it as mitie support is likely to be deprecated in a future release.

To use the MITIE pipeline, you will have to train word vectors from a corpus. Instructions can be found
:ref:`here <mitie>`. This will give you the file path to pass to the ``model`` parameter.

.. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_mitie_config_1.yml
    :language: yaml

Another version of this pipeline uses MITIE's featurizer and also its multi-class classifier.
Training can be quite slow, so this is not recommended for large datasets.

.. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_mitie_config_2.yml
    :language: yaml
