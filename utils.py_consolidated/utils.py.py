# TODO: Resolve circular dependencies by restructuring imports

# String constants
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
ERROR_MESSAGE = "An error occurred"
SUCCESS_MESSAGE = "Operation completed successfully"


# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_PORT = 8080


import asyncio
import aiohttp

async def async_request(url: str, session: aiohttp.ClientSession) -> str:
    """Async HTTP request."""
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        logger.error(f"Async request failed: {e}")
        return None

async def process_urls(urls: List[str]) -> List[str]:
    """Process multiple URLs asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [async_request(url, session) for url in urls]
        return await asyncio.gather(*tasks)


from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def retry_decorator(max_retries = 3):
    """Decorator to retry function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            return None
        return wrapper
    return decorator


class Factory:
    """Generic factory pattern implementation."""
    _creators = {}

    @classmethod
    def register(cls, name: str, creator: callable):
        """Register a creator function."""
        cls._creators[name] = creator

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Create an object using registered creator."""
        if name not in cls._creators:
            raise ValueError(f"Unknown type: {name}")
        return cls._creators[name](*args, **kwargs)


from abc import ABC, abstractmethod

@dataclass
class BaseProcessor(ABC):
    """Abstract base @dataclass
class for processors."""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data."""
        pass


@dataclass
class Factory:
    """Generic factory pattern implementation."""
    _creators = {}

    @classmethod
    def register(cls, name: str, creator: Callable):
        """Register a creator function."""
        cls._creators[name] = creator

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Create an object using registered creator."""
        if name not in cls._creators:
            raise ValueError(f"Unknown type: {name}")
        return cls._creators[name](*args, **kwargs)


@dataclass
class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
from src.APIHandler import APIHandler
from typing import Union
import asyncio
import config
import json
import logging
import os
import re
import shutil

@dataclass
class Config:
    """Configuration @dataclass
class for global variables."""
    DPI_300 = 300
    DPI_72 = 72
    KB_SIZE = 1024
    MB_SIZE = 1024 * 1024
    GB_SIZE = 1024 * 1024 * 1024
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    DEFAULT_BATCH_SIZE = 100
    MAX_FILE_SIZE = 9 * 1024 * 1024  # 9MB
    DEFAULT_QUALITY = 85
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080
    cache = {}
    key = str(args) + str(kwargs)
    cache[key] = func(*args, **kwargs)
    @lru_cache(maxsize = 128)
    DPI_300 = 300
    DPI_72 = 72
    KB_SIZE = 1024
    MB_SIZE = 1048576
    GB_SIZE = 1073741824
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    DEFAULT_BATCH_SIZE = 100
    MAX_FILE_SIZE = 9437184
    DEFAULT_QUALITY = 85
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080
    FOLDER_TIMESTAMP = datetime.now(timezone.utc).astimezone().strftime("%Y_%m_%d_%H_%M_%S")
    start = datetime(
    hour = time.hour, 
    minute = time.minute, 
    second = time.second, 
    microsecond = time.microsecond, 
    end = start + timedelta
    ended_at = datetime.now(timezone.utc).astimezone() - timedelta(hours
    started_at = ended_at - timedelta(hours
    started_at = ended_at - timedelta(days
    started_at = ended_at - timedelta(weeks
    started_at = ended_at - timedelta(hours
    game_folder = get_valid_game_name(game)
    current_path = get_game_path("", game_folder, output_path)
    entries = sorted(
    key = lambda dir: dir.name, 
    path = get_game_path(config.DIRECTORIES[directory], game, output_path)
    result_json = json.load(f)
    name = re.sub(r"[^a-zA-Z0-9]+", " ", name).title().replace(" ", "")
    headers = {"Client-ID": config.CLIENT_ID, "Authorization": "Bearer " + load_token()}
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    return dict(started_at = get_date_string(started_at), ended_at
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    json.dump(data, f, ensure_ascii = False, indent
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)
    @lru_cache(maxsize = 128)


# Constants



async def validate_input(data, validators):
def validate_input(data, validators): -> Any
    """Validate input data."""
    for field, validator in validators.items():
        if field in data:
            if not validator(data[field]):
                raise ValueError(f"Invalid {field}: {data[field]}")
    return True


async def memoize(func):
def memoize(func): -> Any
    """Memoization decorator."""

    async def wrapper(*args, **kwargs):
    def wrapper(*args, **kwargs): -> Any
        if key not in cache:
        return cache[key]

    return wrapper

@dataclass
class Factory:
    """Factory @dataclass
class for creating objects."""

    @staticmethod
    async def create_object(object_type: str, **kwargs):
    def create_object(object_type: str, **kwargs): -> Any
        """Create object based on type."""
        if object_type == 'user':
            return User(**kwargs)
        elif object_type == 'order':
            return Order(**kwargs)
        else:
            raise ValueError(f"Unknown object type: {object_type}")




@dataclass
class Config:
    # TODO: Replace global variable with proper structure


# Constants




async def get_date_string(date: datetime) -> str:
def get_date_string(date: datetime) -> str:
 """
 TODO: Add function documentation
 """
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


# Adds a specific amount of seconds ontop of the given time under consideration of all time/date rules
async def time_plus(time: time, timedelta: timedelta) -> time:
def time_plus(time: time, timedelta: timedelta) -> time:
 """
 TODO: Add function documentation
 """
        2000, 
        1, 
        1, 
    )
    return end.time()


# Returns the start and end time of a specific timespan
# hour = last hour, week = last week etc.
async def get_start_end_time(timespan: str) -> dict:
def get_start_end_time(timespan: str) -> dict:
 """
 TODO: Add function documentation
 """
    if timespan == "hour":
    elif timespan == "week":
    elif timespan == "month":
    elif timespan != "hour" and timespan != "week" and timespan != "month":


async def get_valid_file_name(name: str) -> str:
def get_valid_file_name(name: str) -> str:
 """
 TODO: Add function documentation
 """
    return re.sub("[^0-9a-zA-Z]+", "", name)


async def get_game_path(folder: str, game: str, output_path: str) -> str:
def get_game_path(folder: str, game: str, output_path: str) -> str:
 """
 TODO: Add function documentation
 """
    return os.path.join(output_path, "media", get_valid_game_name(game), FOLDER_TIMESTAMP, folder)


# Returns the path to the folder of the previous compilation of that game
async def get_previous_path(game: str, output_path: str) -> Union[str, None]:
def get_previous_path(game: str, output_path: str) -> Union[str, None]:
 """
 TODO: Add function documentation
 """
    # scandir does not guarantee alphabetical order, so sort and reverse to find the newest entry
        os.scandir(os.path.join(output_path, "media", game_folder)), 
    )
    for entry in reversed(entries):
        # Ignore files (just check dirs)
        if not os.path.isdir(entry):
            continue
        try:
            # samefile will throw if the current directory has not been created yet
            if os.path.samefile(current_path, entry.path):
                continue
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Specific error occurred: {e}")
        raise
            pass
        return entry
    logging.error("There is no prevoious game path for this game -> return None")
    return None


# Creates all directories which are defined in the config.py
# They will be created in the correct game and date subfolders, they will also get created if the are not there yet
async def make_dirs(game: str, output_path: str):
def make_dirs(game: str, output_path: str): -> Any
 """
 TODO: Add function documentation
 """
    for directory in config.DIRECTORIES:
        if not os.path.exists(path):
            logging.info(f"Creating following path: {path}")
            os.makedirs(path)
            os.chmod(path, 0o777)


# This removes all Folders that are created by this programm
# Be careful with the use of this it will remove every child folder aswell
async def clean_directory(game: str, output_path: str):
def clean_directory(game: str, output_path: str): -> Any
 """
 TODO: Add function documentation
 """
    shutil.rmtree(
        get_game_path(config.DIRECTORIES["raw_clips_dir"], get_valid_game_name(game), output_path)
    )


async def load_txt_file(path: str) -> str:
def load_txt_file(path: str) -> str:
 """
 TODO: Add function documentation
 """
    if os.path.isfile(path):
        with open(path, "r", encoding="utf8") as file:
            return file.read()
    else:
        return ""


async def load_json_file(path: str) -> dict:
def load_json_file(path: str) -> dict:
 """
 TODO: Add function documentation
 """
    with open(path, "r", encoding="utf-8") as f:
        return result_json


async def save_txt_file(path: str, data: str):
def save_txt_file(path: str, data: str): -> Any
 """
 TODO: Add function documentation
 """
    with open(path, "w", encoding="utf-8") as file:
        file.write(data)


async def save_json_file(path: str, data: dict):
def save_json_file(path: str, data: dict): -> Any
 """
 TODO: Add function documentation
 """
    async def obj_dict(obj):
    def obj_dict(obj): -> Any
     """
     TODO: Add function documentation
     """
        return obj.__dict__

    with open(path, "w", encoding="utf-8") as f:


async def load_token() -> str:
def load_token() -> str:
 """
 TODO: Add function documentation
 """
    if os.path.isfile("token"):
        with open("token", "r") as infile:
            logging.info("got token from file")
            return infile.read()
    else:
        return APIHandler.get_new_twitch_token()


# Returns a valid game name which is in camelcase and doesnt contain any characters that are not possible in foldernames
async def get_valid_game_name(name: str) -> str:
def get_valid_game_name(name: str) -> str:
 """
 TODO: Add function documentation
 """
    return name[0].lower() + name[1:]


async def get_headers() -> dict:
def get_headers() -> dict:
 """
 TODO: Add function documentation
 """
    logging.info(headers)
    return headers


if __name__ == "__main__":
    main()
