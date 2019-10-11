import os
import tempfile
import time
import shutil
from typing import Text, Optional

import pytest
from _pytest.tmpdir import TempdirFactory

import rasa
import rasa.core
import rasa.nlu
from rasa.importers.rasa import RasaFileImporter
from rasa.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH
from rasa.core.domain import Domain
from rasa.model import (
    FINGERPRINT_CONFIG_KEY,
    FINGERPRINT_DOMAIN_KEY,
    FINGERPRINT_FILE_PATH,
    FINGERPRINT_NLU_DATA_KEY,
    FINGERPRINT_RASA_VERSION_KEY,
    FINGERPRINT_STORIES_KEY,
    FINGERPRINT_TRAINED_AT_KEY,
    core_fingerprint_changed,
    create_package_rasa,
    get_latest_model,
    get_model,
    get_model_subdirectories,
    model_fingerprint,
    nlu_fingerprint_changed,
    Fingerprint,
    should_retrain,
    FINGERPRINT_CONFIG_CORE_KEY,
    FINGERPRINT_CONFIG_NLU_KEY,
)
from rasa.exceptions import ModelNotFound


def test_get_latest_model(trained_model):
    import shutil

    path_of_latest = os.path.join(os.path.dirname(trained_model), "latest.tar.gz")
    shutil.copy(trained_model, path_of_latest)

    model_directory = os.path.dirname(path_of_latest)

    assert get_latest_model(model_directory) == path_of_latest


def test_get_model_from_directory(trained_model):
    unpacked = get_model(trained_model)

    assert os.path.exists(os.path.join(unpacked, "core"))
    assert os.path.exists(os.path.join(unpacked, "nlu"))


def test_get_model_context_manager(trained_model):
    with get_model(trained_model) as unpacked:
        assert os.path.exists(unpacked)

    assert not os.path.exists(unpacked)


@pytest.mark.parametrize("model_path", ["foobar", "rasa", "README.md", None])
def test_get_model_exception(model_path):
    with pytest.raises(ModelNotFound):
        get_model(model_path)


def test_get_model_from_directory_with_subdirectories(
    trained_model, tmpdir_factory: TempdirFactory
):
    unpacked = get_model(trained_model)
    unpacked_core, unpacked_nlu = get_model_subdirectories(unpacked)

    assert unpacked_core
    assert unpacked_nlu

    directory = tmpdir_factory.mktemp("empty_model_dir").strpath
    with pytest.raises(ModelNotFound):
        get_model_subdirectories(directory)


def test_get_model_from_directory_nlu_only(trained_model):
    unpacked = get_model(trained_model)
    shutil.rmtree(os.path.join(unpacked, "core"))
    unpacked_core, unpacked_nlu = get_model_subdirectories(unpacked)

    assert not unpacked_core
    assert unpacked_nlu


def _fingerprint(
    config: Optional[Text] = None,
    config_nlu: Optional[Text] = None,
    config_core: Optional[Text] = None,
    domain: Optional[int] = None,
    rasa_version: Text = "1.0",
    stories: Optional[int] = None,
    nlu: Optional[int] = None,
):
    return {
        FINGERPRINT_CONFIG_KEY: config if config is not None else ["test"],
        FINGERPRINT_CONFIG_CORE_KEY: config_core
        if config_core is not None
        else ["test"],
        FINGERPRINT_CONFIG_NLU_KEY: config_nlu if config_nlu is not None else ["test"],
        FINGERPRINT_DOMAIN_KEY: domain if domain is not None else ["test"],
        FINGERPRINT_TRAINED_AT_KEY: time.time(),
        FINGERPRINT_RASA_VERSION_KEY: rasa_version,
        FINGERPRINT_STORIES_KEY: stories if stories is not None else ["test"],
        FINGERPRINT_NLU_DATA_KEY: nlu if nlu is not None else ["test"],
    }


def test_persist_and_load_fingerprint():
    from rasa.model import persist_fingerprint, fingerprint_from_path

    fingerprint = _fingerprint()
    output_directory = tempfile.mkdtemp()

    persist_fingerprint(output_directory, fingerprint)
    actual = fingerprint_from_path(output_directory)

    assert actual == fingerprint


@pytest.mark.parametrize(
    "fingerprint2",
    [
        _fingerprint(config=["other"]),
        _fingerprint(domain=["other"]),
        _fingerprint(domain=Domain.empty()),
        _fingerprint(stories=["test", "other"]),
        _fingerprint(rasa_version="100"),
        _fingerprint(config=["other"], domain=["other"]),
    ],
)
def test_core_fingerprint_changed(fingerprint2):
    fingerprint1 = _fingerprint()
    assert core_fingerprint_changed(fingerprint1, fingerprint2)


@pytest.mark.parametrize(
    "fingerprint2",
    [
        _fingerprint(config=["other"]),
        _fingerprint(nlu=["test", "other"]),
        _fingerprint(rasa_version="100"),
        _fingerprint(rasa_version="100", config=["other"]),
    ],
)
def test_nlu_fingerprint_changed(fingerprint2):
    fingerprint1 = _fingerprint()
    assert nlu_fingerprint_changed(fingerprint1, fingerprint2)


def _project_files(
    project,
    config_file=DEFAULT_CONFIG_PATH,
    domain="domain.yml",
    training_files=DEFAULT_DATA_PATH,
):
    paths = {
        "config_file": config_file,
        "domain_path": domain,
        "training_data_paths": training_files,
    }

    paths = {k: v if v is None else os.path.join(project, v) for k, v in paths.items()}
    paths["training_data_paths"] = [paths["training_data_paths"]]

    return RasaFileImporter(**paths)


async def test_create_fingerprint_from_paths(project):
    project_files = _project_files(project)

    assert await model_fingerprint(project_files)


@pytest.mark.parametrize(
    "project_files", [["invalid", "invalid", "invalid"], [None, None, None]]
)
async def test_create_fingerprint_from_invalid_paths(project, project_files):
    from rasa.nlu.training_data import TrainingData
    from rasa.core.training.structures import StoryGraph

    project_files = _project_files(project, *project_files)
    expected = _fingerprint(
        config="",
        config_nlu="",
        config_core="",
        domain=hash(Domain.empty()),
        rasa_version=rasa.__version__,
        stories=hash(StoryGraph([])),
        nlu=hash(TrainingData()),
    )

    actual = await model_fingerprint(project_files)
    assert actual[FINGERPRINT_TRAINED_AT_KEY] is not None

    del actual[FINGERPRINT_TRAINED_AT_KEY]
    del expected[FINGERPRINT_TRAINED_AT_KEY]

    assert actual == expected


@pytest.mark.parametrize("use_fingerprint", [True, False])
async def test_rasa_packaging(trained_model, project, use_fingerprint):
    unpacked_model_path = get_model(trained_model)

    os.remove(os.path.join(unpacked_model_path, FINGERPRINT_FILE_PATH))
    if use_fingerprint:
        fingerprint = await model_fingerprint(_project_files(project))
    else:
        fingerprint = None

    tempdir = tempfile.mkdtemp()
    output_path = os.path.join(tempdir, "test.tar.gz")

    create_package_rasa(unpacked_model_path, output_path, fingerprint)

    unpacked = get_model(output_path)

    assert (
        os.path.exists(os.path.join(unpacked, FINGERPRINT_FILE_PATH)) == use_fingerprint
    )
    assert os.path.exists(os.path.join(unpacked, "core"))
    assert os.path.exists(os.path.join(unpacked, "nlu"))

    assert not os.path.exists(unpacked_model_path)


@pytest.mark.parametrize(
    "fingerprint",
    [
        {
            "new": _fingerprint(),
            "old": _fingerprint(stories=["others"]),
            "retrain_core": True,
            "retrain_nlu": False,
        },
        {
            "new": _fingerprint(nlu=["others"]),
            "old": _fingerprint(),
            "retrain_core": False,
            "retrain_nlu": True,
        },
        {
            "new": _fingerprint(config="others"),
            "old": _fingerprint(),
            "retrain_core": True,
            "retrain_nlu": True,
        },
        {
            "new": _fingerprint(config_core="others"),
            "old": _fingerprint(),
            "retrain_core": True,
            "retrain_nlu": False,
        },
        {
            "new": _fingerprint(),
            "old": _fingerprint(config_nlu="others"),
            "retrain_core": False,
            "retrain_nlu": True,
        },
        {
            "new": _fingerprint(),
            "old": _fingerprint(),
            "retrain_core": False,
            "retrain_nlu": False,
        },
    ],
)
def test_should_retrain(trained_model, fingerprint):
    old_model = set_fingerprint(trained_model, fingerprint["old"])

    retrain_core, retrain_nlu = should_retrain(
        fingerprint["new"], old_model, tempfile.mkdtemp()
    )

    assert retrain_core == fingerprint["retrain_core"]
    assert retrain_nlu == fingerprint["retrain_nlu"]


def set_fingerprint(
    trained_model: Text, fingerprint: Fingerprint, use_fingerprint: bool = True
) -> Text:
    unpacked_model_path = get_model(trained_model)

    os.remove(os.path.join(unpacked_model_path, FINGERPRINT_FILE_PATH))
    if use_fingerprint:
        fingerprint = fingerprint
    else:
        fingerprint = None

    tempdir = tempfile.mkdtemp()
    output_path = os.path.join(tempdir, "test.tar.gz")

    create_package_rasa(unpacked_model_path, output_path, fingerprint)

    return output_path
