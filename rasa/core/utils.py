# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import re
import sys
from asyncio import Future
from decimal import Decimal
from hashlib import md5, sha1
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Text,
    Tuple,
    Callable,
    Union,
)

import aiohttp
from aiohttp import InvalidURL
from sanic import Sanic
from sanic.views import CompositionView

import rasa.utils.io as io_utils
from rasa.constants import ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS

# backwards compatibility 1.0.x
# noinspection PyUnresolvedReferences
from rasa.core.lock_store import LockStore, RedisLockStore
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from random import Random


def configure_file_logging(logger_obj: logging.Logger, log_file: Optional[Text]):
    if not log_file:
        return

    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logger_obj.level)
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)


def module_path_from_instance(inst: Any) -> Text:
    """Return the module path of an instance's class."""
    return inst.__module__ + "." + inst.__class__.__name__


def dump_obj_as_json_to_file(filename: Text, obj: Any) -> None:
    """Dump an object as a json string to a file."""

    dump_obj_as_str_to_file(filename, json.dumps(obj, indent=2))


def dump_obj_as_str_to_file(filename: Text, text: Text) -> None:
    """Dump a text to a file."""

    with open(filename, "w", encoding="utf-8") as f:
        # noinspection PyTypeChecker
        f.write(str(text))


def subsample_array(
    arr: List[Any],
    max_values: int,
    can_modify_incoming_array: bool = True,
    rand: Optional["Random"] = None,
) -> List[Any]:
    """Shuffles the array and returns `max_values` number of elements."""
    import random

    if not can_modify_incoming_array:
        arr = arr[:]
    if rand is not None:
        rand.shuffle(arr)
    else:
        random.shuffle(arr)
    return arr[:max_values]


def is_int(value: Any) -> bool:
    """Checks if a value is an integer.

    The type of the value is not important, it might be an int or a float."""

    # noinspection PyBroadException
    try:
        return value == int(value)
    except Exception:
        return False


def one_hot(hot_idx, length, dtype=None):
    import numpy

    if hot_idx >= length:
        raise ValueError(
            "Can't create one hot. Index '{}' is out "
            "of range (length '{}')".format(hot_idx, length)
        )
    r = numpy.zeros(length, dtype)
    r[hot_idx] = 1
    return r


def str_range_list(start, end):
    return [str(e) for e in range(start, end)]


def generate_id(prefix="", max_chars=None):
    import uuid

    gid = uuid.uuid4().hex
    if max_chars:
        gid = gid[:max_chars]

    return "{}{}".format(prefix, gid)


def request_input(valid_values=None, prompt=None, max_suggested=3):
    def wrong_input_message():
        print (
            "Invalid answer, only {}{} allowed\n".format(
                ", ".join(valid_values[:max_suggested]),
                ",..." if len(valid_values) > max_suggested else "",
            )
        )

    while True:
        try:
            input_value = input(prompt) if prompt else input()
            if valid_values is not None and input_value not in valid_values:
                wrong_input_message()
                continue
        except ValueError:
            wrong_input_message()
            continue
        return input_value


# noinspection PyPep8Naming


class HashableNDArray(object):
    """Hashable wrapper for ndarray objects.

    Instances of ndarray are not hashable, meaning they cannot be added to
    sets, nor used as keys in dictionaries. This is by design - ndarray
    objects are mutable, and therefore cannot reliably implement the
    __hash__() method.

    The hashable class allows a way around this limitation. It implements
    the required methods for hashable objects in terms of an encapsulated
    ndarray object. This can be either a copied instance (which is safer)
    or the original object (which requires the user to be careful enough
    not to modify it)."""

    def __init__(self, wrapped, tight=False):
        """Creates a new hashable object encapsulating an ndarray.

        wrapped
            The wrapped ndarray.

        tight
            Optional. If True, a copy of the input ndaray is created.
            Defaults to False.
        """
        from numpy import array

        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view()).hexdigest(), 16)

    def __eq__(self, other):
        from numpy import all

        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        """Returns the encapsulated ndarray.

        If the wrapper is "tight", a copy of the encapsulated ndarray is
        returned. Otherwise, the encapsulated ndarray itself is returned."""
        from numpy import array

        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped


def _dump_yaml(obj, output):
    import ruamel.yaml

    yaml_writer = ruamel.yaml.YAML(pure=True, typ="safe")
    yaml_writer.unicode_supplementary = True
    yaml_writer.default_flow_style = False
    yaml_writer.version = "1.1"

    yaml_writer.dump(obj, output)


def dump_obj_as_yaml_to_file(filename: Union[Text, Path], obj: Dict) -> None:
    """Writes data (python dict) to the filename in yaml repr."""
    with open(str(filename), "w", encoding="utf-8") as output:
        _dump_yaml(obj, output)


def dump_obj_as_yaml_to_string(obj: Dict) -> Text:
    """Writes data (python dict) to a yaml string."""
    str_io = StringIO()
    _dump_yaml(obj, str_io)
    return str_io.getvalue()


def list_routes(app: Sanic):
    """List all the routes of a sanic application.

    Mainly used for debugging."""
    from urllib.parse import unquote

    output = {}

    def find_route(suffix, path):
        for name, (uri, _) in app.router.routes_names.items():
            if name.split(".")[-1] == suffix and uri == path:
                return name
        return None

    for endpoint, route in app.router.routes_all.items():
        if endpoint[:-1] in app.router.routes_all and endpoint[-1] == "/":
            continue

        options = {}
        for arg in route.parameters:
            options[arg] = "[{0}]".format(arg)

        if not isinstance(route.handler, CompositionView):
            handlers = [(list(route.methods)[0], route.name)]
        else:
            handlers = [
                (method, find_route(v.__name__, endpoint) or v.__name__)
                for method, v in route.handler.handlers.items()
            ]

        for method, name in handlers:
            line = unquote("{:50s} {:30s} {}".format(endpoint, method, name))
            output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    logger.debug("Available web server routes: \n{}".format(url_table))

    return output


def cap_length(s, char_limit=20, append_ellipsis=True):
    """Makes sure the string doesn't exceed the passed char limit.

    Appends an ellipsis if the string is too long."""

    if len(s) > char_limit:
        if append_ellipsis:
            return s[: char_limit - 3] + "..."
        else:
            return s[:char_limit]
    else:
        return s


def extract_args(
    kwargs: Dict[Text, Any], keys_to_extract: Set[Text]
) -> Tuple[Dict[Text, Any], Dict[Text, Any]]:
    """Go through the kwargs and filter out the specified keys.

    Return both, the filtered kwargs as well as the remaining kwargs."""

    remaining = {}
    extracted = {}
    for k, v in kwargs.items():
        if k in keys_to_extract:
            extracted[k] = v
        else:
            remaining[k] = v

    return extracted, remaining


def all_subclasses(cls: Any) -> List[Any]:
    """Returns all known (imported) subclasses of a class."""

    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in all_subclasses(s)
    ]


def is_limit_reached(num_messages, limit):
    return limit is not None and num_messages >= limit


def read_lines(filename, max_line_limit=None, line_pattern=".*"):
    """Read messages from the command line and print bot responses."""

    line_filter = re.compile(line_pattern)

    with open(filename, "r", encoding="utf-8") as f:
        num_messages = 0
        for line in f:
            m = line_filter.match(line)
            if m is not None:
                yield m.group(1 if m.lastindex else 0)
                num_messages += 1

            if is_limit_reached(num_messages, max_line_limit):
                break


def file_as_bytes(path: Text) -> bytes:
    """Read in a file as a byte array."""
    with open(path, "rb") as f:
        return f.read()


def convert_bytes_to_string(data: Union[bytes, bytearray, Text]) -> Text:
    """Convert `data` to string if it is a bytes-like object."""

    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8")

    return data


def get_file_hash(path: Text) -> Text:
    """Calculate the md5 hash of a file."""
    return md5(file_as_bytes(path)).hexdigest()


def get_text_hash(text: Text, encoding: Text = "utf-8") -> Text:
    """Calculate the md5 hash for a text."""
    return md5(text.encode(encoding)).hexdigest()


def get_dict_hash(data: Dict, encoding: Text = "utf-8") -> Text:
    """Calculate the md5 hash of a dictionary."""
    return md5(json.dumps(data, sort_keys=True).encode(encoding)).hexdigest()


async def download_file_from_url(url: Text) -> Text:
    """Download a story file from a url and persists it into a temp file.

    Returns the file path of the temp file that contains the
    downloaded content."""
    from rasa.nlu import utils as nlu_utils

    if not nlu_utils.is_url(url):
        raise InvalidURL(url)

    async with aiohttp.ClientSession() as session:
        async with session.get(url, raise_for_status=True) as resp:
            filename = io_utils.create_temporary_file(await resp.read(), mode="w+b")

    return filename


def remove_none_values(obj: Dict[Text, Any]) -> Dict[Text, Any]:
    """Remove all keys that store a `None` value."""
    return {k: v for k, v in obj.items() if v is not None}


def pad_lists_to_size(
    list_x: List, list_y: List, padding_value: Optional[Any] = None
) -> Tuple[List, List]:
    """Compares list sizes and pads them to equal length."""

    difference = len(list_x) - len(list_y)

    if difference > 0:
        return list_x, list_y + [padding_value] * difference
    elif difference < 0:
        return list_x + [padding_value] * (-difference), list_y
    else:
        return list_x, list_y


class AvailableEndpoints(object):
    """Collection of configured endpoints."""

    @classmethod
    def read_endpoints(cls, endpoint_file):
        nlg = read_endpoint_config(endpoint_file, endpoint_type="nlg")
        nlu = read_endpoint_config(endpoint_file, endpoint_type="nlu")
        action = read_endpoint_config(endpoint_file, endpoint_type="action_endpoint")
        model = read_endpoint_config(endpoint_file, endpoint_type="models")
        tracker_store = read_endpoint_config(
            endpoint_file, endpoint_type="tracker_store"
        )
        lock_store = read_endpoint_config(endpoint_file, endpoint_type="lock_store")
        event_broker = read_endpoint_config(endpoint_file, endpoint_type="event_broker")

        return cls(nlg, nlu, action, model, tracker_store, lock_store, event_broker)

    def __init__(
        self,
        nlg=None,
        nlu=None,
        action=None,
        model=None,
        tracker_store=None,
        lock_store=None,
        event_broker=None,
    ):
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.event_broker = event_broker


# noinspection PyProtectedMember
def set_default_subparser(parser, default_subparser):
    """default subparser selection. Call after setup, just before parse_args()

    parser: the name of the parser you're making changes to
    default_subparser: the name of the subparser to call by default"""
    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ["-h", "--help"]:  # global help if no subparser
            break
    else:
        for x in parser._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            # insert default in first position before all other arguments
            sys.argv.insert(1, default_subparser)


def create_task_error_logger(error_message: Text = "") -> Callable[[Future], None]:
    """Error logger to be attached to a task.

    This will ensure exceptions are properly logged and won't get lost."""

    def handler(fut: Future) -> None:
        # noinspection PyBroadException
        try:
            fut.result()
        except Exception:
            logger.exception(
                "An exception was raised while running task. "
                "{}".format(error_message)
            )

    return handler


def replace_floats_with_decimals(obj: Union[List, Dict]) -> Any:
    """
    Utility method to recursively walk a dictionary or list converting all `float` to `Decimal` as required by DynamoDb.

    Args:
        obj: A `List` or `Dict` object.

    Returns: An object with all matching values and `float` type replaced by `Decimal`.

    """
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = replace_floats_with_decimals(obj[i])
        return obj
    elif isinstance(obj, dict):
        for j in obj:
            obj[j] = replace_floats_with_decimals(obj[j])
        return obj
    elif isinstance(obj, float):
        return Decimal(obj)
    else:
        return obj


def _lock_store_is_redis_lock_store(
    lock_store: Union[EndpointConfig, LockStore, None]
) -> bool:
    # determine whether `lock_store` is associated with a `RedisLockStore`
    if isinstance(lock_store, LockStore):
        if isinstance(lock_store, RedisLockStore):
            return True
        return False

    # `lock_store` is `None` or `EndpointConfig`
    return lock_store is not None and lock_store.type == "redis"


def number_of_sanic_workers(lock_store: Union[EndpointConfig, LockStore, None]) -> int:
    """Get the number of Sanic workers to use in `app.run()`.

    If the environment variable constants.ENV_SANIC_WORKERS is set and is not equal to
    1, that value will only be permitted if the used lock store supports shared
    resources across multiple workers (e.g. ``RedisLockStore``).
    """

    def _log_and_get_default_number_of_workers():
        logger.debug(
            f"Using the default number of Sanic workers ({DEFAULT_SANIC_WORKERS})."
        )
        return DEFAULT_SANIC_WORKERS

    try:
        env_value = int(os.environ.get(ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS))
    except ValueError:
        logger.error(
            f"Cannot convert environment variable `{ENV_SANIC_WORKERS}` "
            f"to int ('{os.environ[ENV_SANIC_WORKERS]}')."
        )
        return _log_and_get_default_number_of_workers()

    if env_value == DEFAULT_SANIC_WORKERS:
        return _log_and_get_default_number_of_workers()

    if env_value < 1:
        logger.debug(
            f"Cannot set number of Sanic workers to the desired value "
            f"({env_value}). The number of workers must be at least 1."
        )
        return _log_and_get_default_number_of_workers()

    if _lock_store_is_redis_lock_store(lock_store):
        logger.debug(f"Using {env_value} Sanic workers.")
        return env_value

    logger.debug(
        f"Unable to assign desired number of Sanic workers ({env_value}) as "
        f"no `RedisLockStore` endpoint configuration has been found."
    )
    return _log_and_get_default_number_of_workers()
