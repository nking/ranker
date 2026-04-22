from typing import TypeVar, Generic, Sequence, NamedTuple, Union
import grain
from array_record.python import array_record_module
import weakref
import msgpack

T = TypeVar("T")

class MovieRating(NamedTuple):
    user_id: int
    movie_id: int
    rating: int
    timestamp: int
    
# T = TypeVar("T", bound=Dict[int, int])
class RandomAccessArrayRecordDataSource(
    grain.sources.RandomAccessDataSource, Generic[T]
):
    """
    a grain DataSource for reading random access array_records.  This can be used
    in a dataloader.
    
    example usages:
        datasource = RandomAccessArrayRecordDataSource(ratings_train_uri)
        for record in datasource:
            pass
            
        batch = datasource.__getitems__([100, 100 + batch_size])
        
        note:  can use type parameter, but it won't affect results:
            datasource = RandomAccessArrayRecordDataSource[MovieRating](ratings_train_uri)
            ...
    """
    def __init__(self, uri: str):
        """
        initialize a randome access array_record dataSource that internally uses
        the fast array_record_module.ArrayRecordReader and is able to read
        in batches.
        :param uri: array_record uri.  the array_record is expected to have been
        written for random access with default groups_size:1 and for having used
        serialization with msgpack.
        If you need support for different group_size, this method needs to be edited
        to look for that in a test read of first row, etc. and add logic for it as needed.
        """
        super().__init__(uri)
        # each worker process opens its own instance of the reader only when it first needs to read data
        self.uri = uri
        self._reader = None
        self._finalizer = None
    
    def _ensure_reader(self):
        """Lazy initializer that works across process boundaries."""
        if self._reader is None:
            self._reader = array_record_module.ArrayRecordReader( self.uri)
            # Register the finalizer only once the reader is actually created
            print(f'reader.num_records()={self._reader.num_records()}')
            self._finalizer = weakref.finalize(self, self._close_reader,
                self._reader)
        return self._reader
    
    @staticmethod
    def _close_reader(reader):
        """Static method ensures we don't create a reference cycle back to self."""
        if reader:
            try:
                reader.close()
            except Exception:
                pass
    
    def __getitem__(self, item_index: Union[int, list]) -> T:
        """
        get item or batch of items starting at index item:index.
        The operation is thread-safe.
        :param item_index: index of item w.r.t. the random access array_record
        :return: the item as a dictionary if batch_size is none, else a list
        of the dictionaries from item_index to item+index + batch+size
        """
        if isinstance(item_index, (list, range)):
            return self.__getitems__(item_index)
        b = self._ensure_reader().read([item_index])
        return msgpack.unpackb(b[0], use_list=False)
    
    def __getitems__(self, indices: Sequence[int]) -> Sequence[T]:
        """
        get a batch of items.  NOTE: it is assumed that the sequence does not have
        gaps in it.
        :param indices: a list of sequential indices without gaps between indicies
        :return: a batch of items
        """
        #TODO: replace with vectorized read if one becomes available
        batch_bytes = self._ensure_reader().read(indices)
        b = [msgpack.unpackb(b, use_list=False) for b in batch_bytes] # list of dictionaries
        return b
    
    def __len__(self) -> int:
        return self._ensure_reader().num_records()
    
    def __repr__(self):
        """
        a string representation of the source's configuration, needed for checkpointing.
        it should uniquely identify the data aource.
        :return: unique representation of the source's configuration
        """
        return f"RandomAccessArrayRecordDataSource {self.uri}"
    
    def __getstate__(self):
        """Prepare for pickling (multiprocessing)."""
        state = self.__dict__.copy()
        state['_reader'] = None
        state['_finalizer'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state in worker process."""
        self.__dict__.update(state)
    
    def _close(self):
        """Allow manual closing as well."""
        if self._reader is not None and self._finalizer:
            #  triggers _close_reader(self._reader)
            self._finalizer()
            self._reader = None
            self._finalizer = None
    
    def __del__(self):
        """Invoked by the Garbage Collector."""
        self._close()
