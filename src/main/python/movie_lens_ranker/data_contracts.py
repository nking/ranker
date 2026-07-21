import json
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model
from array_record.python import array_record_module
import msgpack

from movie_lens_ranker.util import uri_access_error

# ==========================  embedding files =========================
class BaseEmbeddingRecord(BaseModel):
    id: int = Field(ge=1, description="Dense integer mapping ID")

    embedding: list[float] = Field(
        min_length=16,
        max_length=16,
        description="16-dimensional float embedding"
    )

class UserEmbeddingRecord(BaseEmbeddingRecord):
    pass

class MovieEmbeddingRecord(BaseEmbeddingRecord):
    id: int = Field(ge=6041, description="Dense integer mapping ID")

# ==========================  movies files =========================
class MovieRecord(BaseModel):
    movie_id: int = Field(ge=6041, description="Dense integer mapping movie ID")
    title: str = Field(min_length=1, description="movie title")
    genres: str = Field(min_length=1, description="genres of movie with '|' for separate.  e.g. Documentary|Comedy")


# ==========================  recommended movies files =========================
class RecommendedMoviesRecord(BaseModel):
    user_id: int = Field(ge=1, description="Dense integer mapping user ID")
    movie_ids: list[int] = Field(min_length=0, max_length=4000, description="movies recommended for user")

class RecommendedMoviesTimestampsRecord(BaseModel):
    user_id: int = Field(ge=1, description="Dense integer mapping user ID")
    #first in MovieLens dataset is Tuesday, April 25, 2000, at 11:05:32 PM UTC == 956703932
    timestamps: list[int] = Field(min_length=0, max_length=4000,
        description="timestamps for recommended movies for user when user has seen movie, else a very large number like the timestamp for 2050 which is 2524608000")

# ==========================  ratings files =========================
class RatingsRecord(BaseModel):
    user_id: int = Field(ge=1, description="Dense integer mapping user ID")
    movie_id: int = Field(ge=6041, description="Dense integer mapping movie ID")
    rating: int = Field(ge=1, le=5, description="rating given to movie by user")
    timestamp: int = Field(ge=956703932, description="timestamp for the rating")


def validate_movies(movies_uri:str, batch_size=256, read_count:int=10):
    adapter = TypeAdapter(list[MovieRecord])
    column_keys = ["movie_id", "title", "genres"]
    _validate_array_record(resource_uri=movies_uri, adapter=adapter, column_keys=column_keys,
                           batch_size=batch_size, read_count=read_count)


def validate_embedding(embedding_uri:str, batch_size=256, read_count:int=10):
    adapter = TypeAdapter(list[BaseEmbeddingRecord])
    column_keys = ["id", "embedding"]
    _validate_array_record(resource_uri=embedding_uri, adapter=adapter, column_keys=column_keys,
                           batch_size=batch_size, read_count=read_count)


def validate_movie_recommendations(movie_recommendations_uri:str, batch_size=256, read_count:int=10):
    adapter = TypeAdapter(list[RecommendedMoviesRecord])
    column_keys = ["user_id", "movie_ids"]
    _validate_array_record(resource_uri=movie_recommendations_uri, adapter=adapter, column_keys=column_keys,
                           batch_size=batch_size, read_count=read_count)


def validate_movie_recommendations_timestamps(movie_rec_ts_uri:str, batch_size=256, read_count:int=10):
    adapter = TypeAdapter(list[RecommendedMoviesTimestampsRecord])
    column_keys = ["user_id", "timestamps"]
    _validate_array_record(resource_uri=movie_rec_ts_uri, adapter=adapter, column_keys=column_keys,
                           batch_size=batch_size, read_count=read_count)


def validate_ratings(ratings_uri:str, batch_size=256, read_count:int=10):
    adapter = TypeAdapter(list[RatingsRecord])
    column_keys = ["user_id", "movie_id", "rating", "timestamp"]
    _validate_array_record(resource_uri=ratings_uri, adapter=adapter, column_keys=column_keys,
                           batch_size=batch_size, read_count=read_count)


def _validate_array_record(resource_uri:str, adapter: TypeAdapter, column_keys: list,
                           batch_size=256, read_count:int=10):
    err = uri_access_error(resource_uri)
    if err is not None:
        raise ValueError(f"{err}")

    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(resource_uri)
        n_records = reader.num_records()
        interval = min(batch_size, read_count)
        end = min(n_records, read_count)
        for i in range(0, end, interval):
            stop = min(i + interval, end)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=True) for b in batch_bytes]

            dict_batch = [
                {key: record[idx] for idx, key in enumerate(column_keys)}
                for record in batch
            ]
            adapter.validate_python(dict_batch)
    finally:
        if reader is not None:
            reader.close()


def create_embedding_schema(embed_len: int):
    """
    Factory function to dynamically construct a Pydantic model for BaseEmbeddingRecord
    with a runtime-configured embedding length.
    """
    return create_model(
        f"EmbeddingRecord_{embed_len}d",
        __base__=BaseEmbeddingRecord,
        embedding=(list[float], Field(min_length=embed_len, max_length=embed_len)),
        id=(int, Field(ge=1))
    )


def export_movie_embedding_contract(output_uri):
    schema = MovieEmbeddingRecord.model_json_schema()
    with open(output_uri, "w") as f:
        json.dump(schema, f, indent=2)

def export_user_embedding_contract(output_uri):
    schema = UserEmbeddingRecord.model_json_schema()
    with open(output_uri, "w") as f:
        json.dump(schema, f, indent=2)

def export_movies_contract(output_uri):
    schema = MovieRecord.model_json_schema()
    with open(output_uri, "w") as f:
        json.dump(schema, f, indent=2)

def export_recommended_movies_contract(output_uri):
    schema = RecommendedMoviesRecord.model_json_schema()
    with open(output_uri, "w") as f:
        json.dump(schema, f, indent=2)

def export_recommended_movies_timestamps_contract(output_uri):
    schema = RecommendedMoviesTimestampsRecord.model_json_schema()
    with open(output_uri, "w") as f:
        json.dump(schema, f, indent=2)

def export_ratings_contract(output_uri):
    schema = RatingsRecord.model_json_schema()
    with open(output_uri, "w") as f:
        json.dump(schema, f, indent=2)
